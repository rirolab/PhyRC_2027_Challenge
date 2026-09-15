"""Opt-in 20 Hz teleop sampling with the existing 60/240 Hz controllers/physics."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .contract import load_contract, pack_measured_state, decode_action, resolve_gripper_intent
from .state import snapshot
from .sensors import RGBDSensors
from .training_dataset import DatasetWriter, keyboard_action
from .evaluation_live import EvaluationSession


class TrainingTeleop:
    def __init__(self, env, cloths, rigs, module, recorder, toggles, output):
        self.env, self.cloths, self.rigs, self.M = env, cloths, rigs, module
        self.recorder, self.toggles = recorder, toggles
        root = Path(__file__).resolve().parents[3]
        if not (root / 'config/policy_interface.json').exists():
            root = Path('/project')
        self.contract = load_contract(root / 'config/policy_interface.json')
        self.runtime = {key: float(getattr(module, key)) for key in
                        ('BASE_LINEAR_RATE', 'BASE_ANGULAR_RATE', 'LIFT_RATE', 'ARM_RATE', 'WRIST_RATE')}
        if not np.isclose(env.world.get_physics_dt(), 1 / 240):
            raise ValueError('Training collection requires the existing 240 Hz physics clock')
        self.sensors = RGBDSensors(env.stage, env.world, rigs, self.contract)
        self.viewport = None
        self.pending_toggles = [0, 0]
        self.phase = 0
        self.inflight = False
        self.current = None
        self.previous = np.zeros((2, 9), np.float32)
        self.targets = []
        metadata = {'raw_archive': str(recorder.archive.path), 'physics_hz': 240, 'policy_hz': 20,
                    'control_hz': 60, 'input_sampling': 'zero-order hold for 3 control ticks / 12 physics ticks',
                    'source_commit': os.environ.get('PHYRC_SOURCE_COMMIT'),
                    'source_dirty': os.environ.get('PHYRC_SOURCE_DIRTY'),
                    'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).resolve().parent.glob('*.py')},
                    'teleop_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                    'contract_sha256': hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest(),
                    'resolved_runtime_rates': self.runtime,
                    'reward_semantics': 'zero placeholder for behavior cloning; scores are audit labels, not actor inputs',
                    'viewport_semantics': 'active GUI camera, same physics state; resized render without UI overlays; audit only'}
        self.writer = DatasetWriter(Path(output) / recorder.archive.path.name, self.contract, metadata)
        self.evaluation = EvaluationSession(env, cloths, rigs, module, self.writer.path / 'evaluation')
        self.closed = False
        print(f'[Training data] START {self.writer.path} | 20Hz sample-and-hold | blocking lossless writes', flush=True)

    def toggle(self, robot_index):
        self.pending_toggles[robot_index] += 1

    def _viewport_product(self):
        import omni.replicator.core as rep
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import UsdGeom, Gf
        viewport = get_active_viewport()
        path = self.M._active_camera_path()
        # Preserve the displayed camera aspect ratio; this is not a screenshot
        # of Kit chrome, collider overlays, or cursor.
        size = tuple(viewport.get_texture_resolution()) if viewport else (256, 256)
        width = 256
        height = max(16, round(width * size[1] / max(1, size[0])))
        # A fixed shape per recording lets HDF5 stream without ragged images.
        if hasattr(self, 'viewport_resolution'):
            width, height = self.viewport_resolution
        else:
            self.viewport_resolution = (width, height)
        if self.viewport is None:
            camera = UsdGeom.Camera.Define(self.env.stage, '/World/TrainingAuditViewport')
            op = camera.AddTransformOp()
            product = rep.create.render_product(str(camera.GetPath()), (width, height))
            rgb = rep.AnnotatorRegistry.get_annotator('rgb')
            rgb.attach([product])
            self.viewport = dict(camera=camera, op=op, product=product, rgb=rgb)
        # Freeze the sampled view during rendering: Kit may dispatch a mouse
        # event while the annotators run, even though physics is not advancing.
        source = UsdGeom.Camera(self.env.stage.GetPrimAtPath(path))
        if not source:
            raise RuntimeError('Active GUI camera unavailable')
        for attr in source.GetPrim().GetAttributes():
            if not attr.GetName().startswith('xformOp') and attr.Get() is not None:
                self.viewport['camera'].GetPrim().CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
        matrix = UsdGeom.Xformable(source).ComputeLocalToWorldTransform(0)
        self.viewport['op'].Set(Gf.Matrix4d(matrix))
        self.viewport['path'] = path

    def capture(self):
        from pxr import UsdGeom
        before = self.recorder.tick
        self._viewport_product()
        values, cameras = self.sensors.capture()
        state, _ = snapshot(self.M, self.cloths, self.rigs)
        obs = pack_measured_state(state, self.contract)
        obs.update(values, previous_action=self.previous.copy())
        if before != self.recorder.tick:
            raise RuntimeError('Rendering advanced physics; dataset synchronization rejected')
        camera_times = np.array([c['timestamp_simulation_s'] for c in cameras], np.float64)
        if not np.allclose(camera_times, float(obs['simulation_time_s']), atol=1e-9, rtol=0):
            raise RuntimeError('Camera and proprioception clocks differ')
        viewport_rgb = np.asarray(self.viewport['rgb'].get_data())
        width, height = self.viewport_resolution
        if viewport_rgb.shape != (height, width, 4):
            raise RuntimeError('Viewport frame unavailable; refusing stale/empty image')
        cam = self.viewport['camera']
        audit = dict(camera_timestamp_s=camera_times,
                     camera_intrinsics=np.array([c['intrinsics'] for c in cameras]),
                     camera_world_from_optical=np.array([c['world_from_optical_column_vectors'] for c in cameras]),
                     viewport_rgb=viewport_rgb[..., :3].copy(),
                     viewport_world=np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(0)),
                     viewport_lens=np.array([cam.GetFocalLengthAttr().Get(), cam.GetHorizontalApertureAttr().Get(),
                                             cam.GetVerticalApertureAttr().Get(), *cam.GetClippingRangeAttr().Get()]),
                     viewport_timestamp_s=np.float64(obs['simulation_time_s']))
        return obs, audit, state

    def before_control(self, held):
        if self.phase == 0:
            if self.current is None:
                self.sensors.reset_episode()
                self.current = self.capture()
                self.writer.start_episode({'start_tick': self.recorder.tick,
                                           'human_spawn': self.env.human_spawn,
                                           'garment_spawn': self.env.garment_spawn,
                                           'joint_names': [[j['name'] for j in r['joints']] for r in self.current[2]['robots']],
                                           'joint_units': [[j['unit'] for j in r['joints']] for r in self.current[2]['robots']],
                                           'seeds': {k: os.environ.get(k) for k in ('HUMAN_SPAWN_SEED', 'STRETCH4_GARMENT_SPAWN_SEED')}})
                self.evaluation.start({'mode': 'training_teleop', 'raw_archive': str(self.recorder.archive.path)})
            # Capture can dispatch new keyboard events. Sample AFTER it, before
            # changing any target, so obs[t] causally precedes action[t].
            self.action = keyboard_action(held, (self.M.ROBOT1_KEYMAP, self.M.ROBOT2_KEYMAP),
                                          [r['state']['gripper_closed'] for r in self.rigs], self.pending_toggles)
            self.pending_toggles = [0, 0]
            self.commands = decode_action(self.action, self.contract, self.runtime)
            self.start_tick = self.recorder.tick
            self.targets = []
            self.inflight = True
            self.recorder.event('policy_action', action=self.action.tolist(), end_tick=self.start_tick + 12,
                                dataset=str(self.writer.path), sample_index=int(self.writer.data.attrs['total']))
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
            raise RuntimeError(f'Teleop control advanced to tick {self.recorder.tick}, expected {expected}; no mislabeled data saved')
        self.phase += 1
        if self.phase != 3:
            return
        self.previous = self.action.copy()
        following = self.capture()
        if not self.evaluation.active:
            raise RuntimeError(f'Training evaluation failed: {self.evaluation.last_report}')
        score = self.evaluation.scorer.result()
        # Scores and all geometry-derived labels live exclusively in audit.
        audit = dict(self.current[1])
        audit.update({'next_' + k: v for k, v in following[1].items()})
        audit.update(controller_targets_applied=np.stack(self.targets),
                     control_start_ticks=np.arange(self.start_tick, self.start_tick + 12, 4, dtype=np.int64),
                     score_items=np.array([item['points'] for item in score['score_items'].values()], np.float32),
                     first_contact_tick=np.int64(-1 if self.evaluation.clock.first_contact_tick is None else
                                                self.evaluation.clock.first_contact_tick),
                     raw_points=np.float32(score['raw_points']))
        audit['score_time_s'] = np.float64(self.evaluation.clock_result(score)['task_time_s'])
        self.writer.append(self.current[0], self.action, following[0], start_tick=self.start_tick,
                           end_tick=self.recorder.tick, audit=audit)
        self.current = following
        self.phase = 0
        self.inflight = False

    def boundary(self, reason, valid=True):
        if self.inflight:
            valid = False
        result = self.evaluation.finish(reason=reason, valid=valid, take_final=valid)
        self.writer.finish_episode(reason, result, valid=valid)
        self.current = None
        self.phase = 0
        self.inflight = False
        self.previous.fill(0)
        self.pending_toggles = [0, 0]

    def close(self, complete=True):
        if self.closed:
            return
        complete = complete and not self.inflight
        try:
            self.boundary('gui_closed' if complete else 'exception', valid=complete and not self.inflight)
        except BaseException:
            complete = False
            raise
        finally:
            self.writer.close(complete=complete)
            self.closed = True
        print(f'[Training data] SAVED {self.writer.path} | complete={complete}', flush=True)
        self.sensors.close()
        if self.viewport:
            self.viewport['rgb'].detach([self.viewport['product']])
            self.viewport['product'].destroy()
