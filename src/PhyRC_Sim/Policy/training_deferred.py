"""Record policy boundaries without rendering; generate RGBD after teleop exits."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .contract import load_contract, pack_measured_state, decode_action, resolve_gripper_intent
from .state import snapshot, array
from .sensors import look_at, pose_matrix
from .training_dataset import keyboard_action
from .teleop_recording import atomic_json


class DeferredTrainingTeleop:
    def __init__(self, env, cloths, rigs, module, recorder, toggles, output):
        from omni.kit.viewport.utility import get_active_viewport
        self.env, self.cloths, self.rigs, self.M = env, cloths, rigs, module
        self.recorder, self.toggles = recorder, toggles
        root = Path(__file__).resolve().parents[3]
        if not (root / 'config/policy_interface.json').exists():
            root = Path('/project')
        self.contract = load_contract(root / 'config/policy_interface.json')
        if not np.isclose(env.world.get_physics_dt(), 1 / 240):
            raise ValueError('Training collection requires 240Hz physics')
        self.runtime = {key: float(getattr(module, key)) for key in
                        ('BASE_LINEAR_RATE', 'BASE_ANGULAR_RATE', 'LIFT_RATE', 'ARM_RATE', 'WRIST_RATE')}
        self.path = Path(output) / recorder.archive.path.name
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {'schema': 'phyrc-deferred-capture-v1', 'contract': self.contract,
                         'raw_archive': str(recorder.archive.path), 'dataset_path': str(self.path),
                         'source_commit': os.environ.get('PHYRC_SOURCE_COMMIT'),
                         'source_dirty': os.environ.get('PHYRC_SOURCE_DIRTY'),
                         'resolved_runtime_rates': self.runtime,
                         'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in Path(__file__).resolve().parent.glob('*.py')},
                         'visual_source': 'rendered after collection from recorded state and recorded camera poses',
                         'reward_semantics': 'zero BC placeholder; scores are audit labels'}
        recorder.event('training_capture', metadata=self.metadata)
        atomic_json(self.path / 'capture.json', dict(self.metadata, status='recording', complete=False))
        viewport = get_active_viewport()
        size = tuple(viewport.get_texture_resolution()) if viewport else (256, 256)
        self.viewport_resolution = np.array([256, max(16, round(256 * size[1] / max(size[0], 1)))], np.int32)
        self.sensors = None  # Explicitly no render products in the interaction process.
        self.pending_toggles = [0, 0]
        self.previous = np.zeros((2, 9), np.float32)
        self.phase, self.inflight, self.closed = 0, False, False
        self.episode, self.frame, self.samples = -1, 0, 0
        self.active = False
        self.started = None
        print(f'[Training data] FAST capture {self.path}; RGBD/HDF5 generated after exit', flush=True)

    def toggle(self, index):
        self.pending_toggles[index] += 1

    def _camera_poses(self):
        from pxr import UsdGeom
        result = []
        for spec in self.contract['cameras']:
            if spec.get('mount') in ('world', 'human_spawn'):
                pose = look_at(np.array(spec['eye_world_m'], float), spec['target_world_m'], [0, 0, 1])
                if spec['mount'] == 'human_spawn':
                    prim = self.env.stage.GetPrimAtPath(spec['spawn_prim_path'])
                    delta = np.eye(4)
                    restored = prim.GetAttribute('phyrc:cameraSpawnDelta')
                    if restored and restored.HasAuthoredValueOpinion():
                        delta = np.array(restored.Get()).T
                    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                        if op.GetOpName() == 'xformOp:transform:randomSpawn':
                            delta = np.array(op.Get()).T
                            break
                    pose = delta @ pose
            else:
                rig = self.rigs[int(spec['id'].split('_')[1])]
                view = rig['robot']._articulation_view
                index = view.get_link_index(spec['mount_role'])
                if index is None or index < 0:
                    raise ValueError('Missing camera mount ' + spec['mount_role'])
                raw = array(view._physics_view.get_link_transforms())[0, index]
                pose = pose_matrix(raw) @ look_at(np.array(spec['eye_mount_m'], float),
                                                spec['target_mount_m'], spec['up_mount'])
            result.append(pose)
        return np.stack(result)

    def _capture(self, targets=None):
        tick = self.recorder.tick
        state, _ = snapshot(self.M, self.cloths, self.rigs)
        obs = pack_measured_state(state, self.contract)
        obs['previous_action'] = self.previous.copy()
        extra = {'policy_obs/' + k: v for k, v in obs.items()}
        extra.update(policy_episode=np.int64(self.episode), policy_frame=np.int64(self.frame),
                     policy_camera_world=self._camera_poses(), policy_viewport_resolution=self.viewport_resolution,
                     policy_applied_targets=np.empty((0, 2, 9), np.float32) if targets is None else np.stack(targets))
        if self.frame == 0:
            extra['evaluation_context'] = self.recorder._evaluation_context
        if tick != self.recorder.tick:
            raise RuntimeError('Physics advanced during observation capture')
        self.recorder.capture('policy_observation', extra)
        return state

    def before_control(self, held):
        if self.phase == 0:
            if not self.active:
                self.episode += 1
                self.frame = 0
                state = self._capture()
                self.active = True
                if self.started is None:
                    self.started = time.monotonic()
                self.recorder.event('policy_episode_start', episode=self.episode,
                                    joint_names=[[j['name'] for j in r['joints']] for r in state['robots']],
                                    joint_units=[[j['unit'] for j in r['joints']] for r in state['robots']],
                                    human_spawn=self.env.human_spawn, garment_spawn=self.env.garment_spawn)
            self.action = keyboard_action(held, (self.M.ROBOT1_KEYMAP, self.M.ROBOT2_KEYMAP),
                                          [r['state']['gripper_closed'] for r in self.rigs], self.pending_toggles)
            self.pending_toggles = [0, 0]
            self.commands = decode_action(self.action, self.contract, self.runtime)
            self.start_tick = self.recorder.tick
            self.targets = []
            self.inflight = True
            self.recorder.event('policy_action', action=self.action.tolist(), end_tick=self.start_tick + 12,
                                dataset=str(self.path), sample_index=self.samples)
            for rig, command, toggle in zip(self.rigs, self.commands, self.toggles):
                closed = bool(rig['state']['gripper_closed'])
                if resolve_gripper_intent(closed, command['gripper_intent']) != closed:
                    toggle()
        return self.commands

    def after_drive(self):
        self.targets.append(np.array([[r['state'][key] for key in self.contract['controller_target_columns']]
                                      for r in self.rigs], np.float32))

    def after_control(self):
        expected = self.start_tick + (self.phase + 1) * 4
        if self.recorder.tick != expected:
            raise RuntimeError(f'Teleop control reached tick {self.recorder.tick}, expected {expected}')
        self.phase += 1
        if self.phase == 3:
            self.previous = self.action.copy()
            self.frame += 1
            self._capture(self.targets)
            self.samples += 1
            self.phase, self.inflight = 0, False

    def boundary(self, reason, valid=True):
        if self.active:
            self.recorder.event('policy_episode_end', episode=self.episode, reason=reason,
                                valid=bool(valid and not self.inflight), samples=self.frame)
        self.active = False
        self.phase, self.inflight = 0, False
        self.previous.fill(0)
        self.pending_toggles = [0, 0]

    def close(self, complete=True):
        if self.closed:
            return
        complete = bool(complete and not self.inflight)
        self.boundary('gui_closed' if complete else 'exception', valid=complete)
        self.closed = True
        seconds = 0 if self.started is None else time.monotonic() - self.started
        atomic_json(self.path / 'capture.json', dict(self.metadata, status='awaiting_export' if complete else 'incomplete',
                    complete=complete, samples=self.samples, collection_wall_s=seconds,
                    collection_simulation_s=self.samples * .05))
        request = os.environ.get('STRETCH4_TRAINING_EXPORT_REQUEST')
        if complete and request:
            atomic_json(Path(request), {'recording': str(self.recorder.archive.path), 'output': str(self.path)})
        print(f'[Training data] CAPTURE SAVED {self.path} | {self.samples} transitions | '
              f'{seconds:.3f}s collection wall time | complete={complete}; images pending export', flush=True)
