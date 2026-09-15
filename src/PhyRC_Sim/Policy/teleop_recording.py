"""Lossless physics-step state archive and teleop event journal.

Replay uses recorded state, not a new FEM solve. Solver internals are not
serialized and this is not a bitwise deterministic simulation checkpoint.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
import queue
import threading
from pathlib import Path
import time
import uuid
import zipfile

import numpy as np

from .state import array


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class StateArchive:
    """Bounded-memory, lossless chunks. Disk backpressure never drops frames."""
    def __init__(self, path, metadata, chunk_frames=16, queue_chunks=4):
        if chunk_frames < 1 or queue_chunks < 1:
            raise ValueError('Chunk and queue sizes must be positive')
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.chunk_frames = chunk_frames
        self.manifest = {'schema': 'phyrc-full-teleop-v2', 'complete': False,
                         'metadata': metadata, 'chunks': [], 'frames': 0}
        self.manifest['storage'] = {'compression': 'lossless DEFLATE level 1',
                                    'chunk_frames': chunk_frames, 'queue_chunks': queue_chunks,
                                    'overflow': 'block; never drop frames'}
        self.buffer = {}
        self.headers = []
        self.count = 0
        self.closed = False
        self.events = (self.path / 'events.jsonl').open('w', encoding='utf-8')
        self.event_sequence = 0
        atomic_json(self.path / 'manifest.json', self.manifest)
        self._queue = queue.Queue(maxsize=queue_chunks)
        self._failure = None
        self._worker = threading.Thread(target=self._write_loop, name='teleop-archive-writer', daemon=True)
        self._worker.start()

    def check(self):
        if self._failure is not None:
            raise RuntimeError(f'Archive writer failed: {self._failure}') from self._failure

    def _write_loop(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._failure is None:
                    self._write_chunk(*item)
            except BaseException as exc:
                self._failure = exc
            finally:
                self._queue.task_done()

    def event(self, tick, kind, **data):
        self.check()
        self.events.write(json.dumps(dict(sequence=self.event_sequence, tick=tick,
                                          kind=kind, dispatch_monotonic_ns=time.monotonic_ns(), **data),
                                     allow_nan=False) + '\n')
        self.events.flush()
        self.event_sequence += 1

    def append(self, tick, kind, values):
        self.check()
        if self.closed:
            raise RuntimeError('Archive already closed')
        for key, value in values.items():
            value = np.asarray(value)
            if value.dtype.kind == 'O':
                raise ValueError('Object/pickle arrays are not permitted')
            if value.dtype.kind in 'fc' and not np.isfinite(value).all():
                raise ValueError(f'Nonfinite recording field: {key}')
            self.buffer[f'f{len(self.headers):04d}/{key}'] = value.copy()
        self.headers.append({'frame': self.count, 'tick': tick, 'kind': kind})
        self.count += 1
        if len(self.headers) >= self.chunk_frames:
            self.flush()

    def flush(self):
        self.check()
        if not self.headers:
            return
        # Transfer ownership. The producer never touches queued arrays again.
        self._queue.put((self.buffer, self.headers))
        self.buffer, self.headers = {}, []
        self.check()

    def sync(self):
        """Commit everything submitted so far (also useful for recovery checks)."""
        self.flush()
        self._queue.join()
        self.check()

    def _write_chunk(self, buffer, headers):
        name = f'chunk_{len(self.manifest["chunks"]):06d}.npz'
        temporary = self.path / (name + '.tmp')
        # Stack equal-shape fields once per chunk. Ragged grasps and optional
        # fields keep their per-frame representation; no casting/quantization.
        packed, stacked = {}, []
        keys = [key.split('/', 1)[1] for key in buffer if key.startswith('f0000/')]
        for key in keys:
            names = [f'f{i:04d}/{key}' for i in range(len(headers))]
            values = [buffer.get(name) for name in names]
            first = values[0]
            if all(v is not None and v.shape == first.shape and v.dtype == first.dtype for v in values):
                packed['stack/' + key] = np.stack(values, dtype=first.dtype, casting='no')
                stacked.append(key)
                for field in names:
                    del buffer[field]
        packed.update(buffer)
        packed['_headers'] = np.array(json.dumps(headers))
        packed['_stacked'] = np.array(json.dumps(stacked))
        with temporary.open('wb') as stream:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for key, value in packed.items():
                    with archive.open(key + '.npy', 'w', force_zip64=True) as entry:
                        np.lib.format.write_array(entry, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        path = self.path / name
        temporary.replace(path)
        self.manifest['chunks'].append({'file': name, 'first_frame': headers[0]['frame'],
                                        'last_frame': headers[-1]['frame'],
                                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        self.manifest['frames'] = headers[-1]['frame'] + 1
        atomic_json(self.path / 'manifest.json', self.manifest)

    def close(self, complete=True, reason='normal_exit'):
        if self.closed:
            return
        try:
            self.flush()
        except Exception as exc:
            self._failure = self._failure or exc
        self._queue.put(None)
        self._worker.join()
        if self._failure is not None:
            complete, reason = False, f'writer_error: {self._failure}'
        self.events.flush()
        os.fsync(self.events.fileno())
        self.events.close()
        self.manifest.update(complete=complete, reason=reason, events=self.event_sequence,
                             events_sha256=hashlib.sha256((self.path / 'events.jsonl').read_bytes()).hexdigest())
        atomic_json(self.path / 'manifest.json', self.manifest)
        self.closed = True
        self.check()


class ArchiveReader:
    def __init__(self, path, allow_incomplete=False):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest.get('schema') not in ('phyrc-full-teleop-v1', 'phyrc-full-teleop-v2'):
            raise ValueError('Unsupported recording schema')
        if not self.manifest.get('complete') and not allow_incomplete:
            raise ValueError('Recording is incomplete; use --allow-incomplete to inspect committed chunks')
        event_hash = self.manifest.get('events_sha256')
        if event_hash and hashlib.sha256((self.path / 'events.jsonl').read_bytes()).hexdigest() != event_hash:
            raise ValueError('Recorded input-event checksum mismatch')

    def frames(self):
        expected, previous_tick = 0, 0
        for chunk in self.manifest['chunks']:
            path = self.path / chunk['file']
            if path.parent.resolve() != self.path.resolve() or hashlib.sha256(path.read_bytes()).hexdigest() != chunk['sha256']:
                raise ValueError('Recording chunk path/checksum mismatch')
            with np.load(path, allow_pickle=False) as data:
                headers = json.loads(str(data['_headers']))
                stacked = ({key: data['stack/' + key] for key in json.loads(str(data['_stacked']))}
                           if self.manifest['schema'] == 'phyrc-full-teleop-v2' else {})
                for i, header in enumerate(headers):
                    tick = header['tick']
                    if header['frame'] != expected or tick < previous_tick or tick > previous_tick + 1:
                        raise ValueError('Recording contains a frame/physics-step gap')
                    if header['kind'] == 'physics' and tick != previous_tick + 1:
                        raise ValueError('Duplicate physics tick in recording')
                    prefix = f'f{i:04d}/'
                    values = {key: value[i] for key, value in stacked.items()}
                    values.update({key[len(prefix):]: data[key] for key in data.files if key.startswith(prefix)})
                    yield header, values
                    expected += 1
                    previous_tick = tick
        if expected != self.manifest['frames']:
            raise ValueError('Recording frame count mismatch')


class FullTeleopRecorder:
    def __init__(self, backend, cloths, rigs, module, output='/output/full_teleop'):
        from pxr import Usd, UsdGeom, UsdPhysics
        self.backend, self.cloths, self.rigs, self.M = backend, cloths, rigs, module
        self.stage = backend.stage
        self.tick = 0
        self.suspended = False
        self.failure = None
        self.subscription = None
        self.dt = float(backend.world.get_physics_dt())
        links, scales = [], []
        for rig in rigs:
            view = rig['robot']._articulation_view
            root = rig['robot'].prim_path.rsplit('/', 1)[0]
            bodies = {p.GetName(): p for p in Usd.PrimRange(self.stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
                      if p.HasAPI(UsdPhysics.RigidBodyAPI)}
            names = list(view.body_names)
            paths = [str(bodies[name].GetPath()) for name in names]
            links.append(paths)
            scales.append([np.linalg.norm(np.asarray(UsdGeom.Xformable(bodies[name]).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()))[:3, :3], axis=1).tolist() for name in names])
        self.control_names = ['lift', 'arm', 'yaw', 'pitch', 'roll', 'grip_pos',
                              'base_fwd', 'base_strafe', 'base_turn', 'gripper_closed']
        metadata = {'physics_dt_s': self.dt, 'physics_hz': 1 / self.dt,
                    'time_basis': 'integer physics tick; duplicate ticks are ordered discontinuity/control boundaries',
                    'replay_mode': 'authoritative 3D state playback; no FEM re-simulation',
                    'robot_link_paths': links, 'robot_link_scales': scales,
                    'grasp_link_indices': [int(r['grasp_link_idx']) for r in rigs],
                    'evaluation_state_version': 1,
                    'joint_names': [list(r['robot'].dof_names) for r in rigs],
                    'garment_paths': [str(c.prim.GetPath()) for c in cloths],
                    'control_names': self.control_names,
                    'source_commit': os.environ.get('PHYRC_SOURCE_COMMIT'),
                    'teleop_source_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                    'human_spawn': backend.human_spawn, 'garment_spawn': backend.garment_spawn,
                    'geometry_revision': module._STATE_GEOMETRY_REVISION,
                    'overrides': {k: v for k, v in os.environ.items() if k.startswith(('STRETCH4_', 'HUMAN_', 'BOX_', 'ROBOT_'))}}
        path = Path(output) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:6])
        self.archive = StateArchive(path, metadata)
        # Flatten keeps the original scene/mesh/material topology for playback.
        # Textures remain resolved asset paths in the same project/container.
        self.stage.Flatten().Export(str(path / 'scene.usdc'))
        self.archive.manifest['scene_sha256'] = hashlib.sha256((path / 'scene.usdc').read_bytes()).hexdigest()
        self._update_evaluation_context()
        self.capture('initial')
        self.subscription = backend.world._physics_context._physics_sim_interface.subscribe_physics_on_step_events(
            pre_step=False, order=300, on_update=self._physics_step)
        print(f'[Full record] START {path} | every physics step, no dropped states', flush=True)

    def _update_evaluation_context(self):
        from .evaluation_live import IsaacMeasurements
        from .evaluation_replay import measurement_context
        measurement = IsaacMeasurements(self.backend, self.cloths, self.rigs, self.M)
        self._evaluation_context = np.array(json.dumps(measurement_context(measurement), allow_nan=False))

    def capture(self, kind, extra=None):
        from pxr import Usd, UsdGeom
        values = {'simulation_time_s': np.array(float(self.backend.world.current_time)),
                  'capture_monotonic_ns': np.array(time.monotonic_ns(), dtype=np.int64)}
        if kind == 'initial' or kind.startswith('after_'):
            values['evaluation_context'] = self._evaluation_context
        for i, cloth in enumerate(self.cloths):
            values[f'g{i}_positions'] = array(cloth.get_world_positions())[0]
            values[f'g{i}_velocities'] = array(cloth.get_velocities())[0]
        for i, rig in enumerate(self.rigs):
            robot, state = rig['robot'], rig['state']
            values[f'r{i}_links'] = array(robot._articulation_view._physics_view.get_link_transforms())[0]
            values[f'r{i}_joint_positions'] = array(robot.get_joint_positions())
            values[f'r{i}_joint_velocities'] = array(robot.get_joint_velocities())
            values[f'r{i}_root_linear_velocity'] = array(robot.get_linear_velocity())
            values[f'r{i}_root_angular_velocity'] = array(robot.get_angular_velocity())
            values[f'r{i}_controls'] = np.array([state.get(k, 0) for k in self.control_names], dtype=np.float64)
            grabbed = state.get('grabbed')
            values[f'r{i}_grab_cloth'] = np.array(-1 if grabbed is None else int(grabbed[0]))
            values[f'r{i}_grab_indices'] = np.array([], dtype=np.int64) if grabbed is None else array(grabbed[1])
            values[f'r{i}_grab_offsets'] = np.empty((0, 3)) if grabbed is None else array(grabbed[2])
            mask, local = state.get('_grab_anchor_mask'), state.get('_native_local_offsets')
            root = robot.prim_path.rsplit('/', 1)[0]
            values[f'r{i}_attachment_present'] = np.array(bool(self.stage.GetPrimAtPath(root + '/ClothGraspAttachment')))
            values[f'r{i}_anchor_mask'] = np.array([], dtype=bool) if mask is None else array(mask)
            values[f'r{i}_anchor_local_offsets'] = np.empty((0, 3)) if local is None else array(local)
            values[f'r{i}_anchor_data_present'] = np.array(mask is not None and local is not None)
        for name in ('Human', 'Chair'):
            values[f'static_{name}'] = np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath('/World/' + name))
                                                 .ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        camera_path = self.M._active_camera_path()
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(camera_path))
        if camera:
            values['camera_world'] = np.asarray(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            values['camera_lens'] = np.array([camera.GetFocalLengthAttr().Get(), camera.GetHorizontalApertureAttr().Get(),
                                              camera.GetVerticalApertureAttr().Get(), *camera.GetClippingRangeAttr().Get()])
        if extra:
            if set(extra) & set(values):
                raise ValueError('Extra capture fields overwrite raw physics state')
            values.update(extra)
        self.archive.append(self.tick, kind, values)

    def _physics_step(self, dt, context=None):
        if self.suspended or self.failure:
            return
        try:
            # PhysX callback dt is float32; World exposes the same rate as a
            # Python double. Compare in the API's precision, never sum dt.
            if np.float32(dt) != np.float32(self.dt):
                raise ValueError(f'Physics dt changed during full recording: {dt} vs {self.dt}')
            self.tick += 1
            self.capture('physics')
        except Exception as exc:
            self.failure = exc

    def check(self):
        self.archive.check()
        if self.failure:
            raise RuntimeError(f'Full recording failed; simulation stopped to avoid an unrecorded interval: {self.failure}') from self.failure

    def event(self, kind, **data):
        self.archive.event(self.tick, kind, **data)

    def begin_discontinuity(self, reason):
        self.event('discontinuity_begin', reason=reason)
        self.capture('before_' + reason)
        self.suspended = True  # PhysX views do not exist while reset/load rebuilds them.

    def end_discontinuity(self, reason):
        self.suspended = False
        self._update_evaluation_context()
        self.capture('after_' + reason)
        self.event('discontinuity_end', reason=reason)

    def close(self, reason='normal_exit', capture_final=True):
        self.subscription = None
        complete = self.failure is None and reason == 'normal_exit'
        if complete and capture_final:
            try:
                self.capture('final')
            except Exception as exc:
                complete, reason = False, str(exc)
        self.archive.close(complete=complete, reason=reason if self.failure is None else str(self.failure))
        print(f'[Full record] SAVED {self.archive.path} | {self.archive.count} states | complete={complete}', flush=True)
