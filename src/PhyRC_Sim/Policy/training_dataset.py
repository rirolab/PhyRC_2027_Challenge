"""Synchronous public observation/action datasets; HDF5, no Isaac imports.

The audit file is deliberately separate from policy inputs. Writes block rather
than dropping samples; an interrupted file remains explicitly incomplete.
"""
import json
from pathlib import Path

import numpy as np

from .contract import validate_observation


def keyboard_action(held, keymaps, gripper_closed, toggles):
    """Sample once per policy interval. Same signs as drive_robot's key maps."""
    names = ('base_fwd', 'base_strafe', 'base_turn', 'lift', 'arm', 'yaw', 'pitch', 'roll')
    action = np.zeros((len(keymaps), 9), np.float32)
    for i, keys in enumerate(keymaps):
        for j, name in enumerate(names):
            action[i, j] = int(keys[name + '_pos'] in held) - int(keys[name + '_neg'] in held)
        action[i, 1] *= -1  # backend strafe is right-positive; policy is left-positive.
        if toggles[i] % 2:
            action[i, 8] = -1 if gripper_closed[i] else 1
    return action


def training_observation(values, contract):
    """Strict allowlist; split cameras and flatten proprioception for robomimic."""
    dims = contract['default_dimensions']
    validate_observation(values, contract, robots=dims['robots'], joints=dims['joints'], profile='actor_rgbd')
    out = {}
    for name, value in values.items():
        if name in ('rgb', 'depth', 'depth_valid'):
            for i, spec in enumerate(contract['cameras']):
                out[spec['id'] + '_' + name] = value[i]
        else:
            out[name] = value.reshape(-1)
    return out


class DatasetWriter:
    def __init__(self, path, contract, metadata):
        import h5py
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.contract = contract
        self.policy = h5py.File(self.path / 'policy.hdf5', 'w')
        self.audit = h5py.File(self.path / 'audit.hdf5', 'w')
        self.data = self.policy.create_group('data')
        self.data.attrs['total'] = 0
        self.data.attrs['env_args'] = json.dumps({'env_name': 'PhyRCDressing', 'type': 2,
                                                'env_kwargs': {'profile': 'actor_rgbd'}})
        self.policy.attrs.update(schema='phyrc-policy-dataset-v1', complete=False,
                                 contract_json=json.dumps(contract))
        self.audit.attrs.update(schema='phyrc-policy-audit-v1', complete=False,
                                metadata_json=json.dumps(metadata, allow_nan=False))
        self.episode = None
        self.closed = False

    def start_episode(self, metadata):
        if self.episode is not None:
            raise RuntimeError('Finish current episode first')
        name = 'demo_' + str(len(self.data))
        self.episode = self.data.create_group(name)
        self.episode.create_group('obs')
        self.episode.create_group('next_obs')
        self.audit_episode = self.audit.create_group(name)
        self.audit_episode.attrs['metadata_json'] = json.dumps(metadata, allow_nan=False)
        self.episode.attrs.update(num_samples=0, complete=False)
        self.last_tick = None
        self.last_time_s = None

    @staticmethod
    def _append(group, name, value):
        value = np.asarray(value)
        if value.dtype.kind == 'O' or (value.dtype.kind in 'fc' and not np.isfinite(value).all()):
            raise ValueError('Invalid dataset value: ' + name)
        if name not in group:
            group.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                 chunks=(1, *value.shape), dtype=value.dtype, compression='lzf')
        dataset = group[name]
        if dataset.shape[1:] != value.shape or dataset.dtype != value.dtype:
            raise ValueError('Dataset shape/dtype changed: ' + name)
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    def append(self, obs, action, next_obs, *, start_tick, end_tick, audit):
        if self.episode is None:
            raise RuntimeError('Start an episode before appending')
        if end_tick - start_tick != 12 or (self.last_tick is not None and start_tick != self.last_tick):
            raise ValueError('Missing or misaligned physics ticks; expected contiguous 12-tick transitions')
        if self.last_time_s is not None and float(obs['simulation_time_s']) != self.last_time_s:
            raise ValueError('Observation clock is discontinuous inside episode')
        action = np.asarray(action)
        if action.shape != (2, 9) or action.dtype != np.float32 or not np.isfinite(action).all() or np.any(abs(action) > 1):
            raise ValueError('Expected normalized float32 (2,9) action')
        if not np.isclose(float(next_obs['simulation_time_s']) - float(obs['simulation_time_s']), .05, atol=1e-7, rtol=0):
            raise ValueError('Observation clock does not match the action interval')
        if not np.array_equal(next_obs['previous_action'], action):
            raise ValueError('next_obs.previous_action differs from applied action')
        packed = [training_observation(v, self.contract) for v in (obs, next_obs)]
        for group, values in zip(('obs', 'next_obs'), packed):
            for key, value in values.items():
                self._append(self.episode[group], key, value)
        for name, value in dict(actions=action.reshape(18), rewards=np.float32(0),
                                dones=np.uint8(0), terminated=np.uint8(0), truncated=np.uint8(0)).items():
            self._append(self.episode, name, value)
        for name, value in dict(start_tick=np.int64(start_tick), end_tick=np.int64(end_tick), **audit).items():
            self._append(self.audit_episode, name, value)
        self.episode.attrs['num_samples'] += 1
        self.data.attrs['total'] += 1
        self.last_tick = end_tick
        self.last_time_s = float(next_obs['simulation_time_s'])
        self.policy.flush()
        self.audit.flush()

    def finish_episode(self, reason, result=None, valid=True):
        if self.episode is None:
            return
        n = int(self.episode.attrs['num_samples'])
        if n:
            self.episode['dones'][-1] = 1
            self.episode['truncated'][-1] = 1
        self.episode.attrs.update(complete=bool(valid), termination_reason=reason)
        self.audit_episode.attrs['result_json'] = json.dumps(result, allow_nan=False)
        score = (result or {}).get('result') or {}
        known = bool(result and result.get('valid') and score.get('success_evaluated', 'dressing_complete' in score))
        self.audit_episode.attrs['success_known'] = known
        self.audit_episode.attrs['success'] = bool(known and score.get('success', score.get('dressing_complete', False)))
        self.audit_episode.attrs['success_basis'] = score.get('success_basis', 'confirmed sleeves and neck for 0.5s; achieved during episode, not raw_points')
        self.episode = None
        self.policy.flush()
        self.audit.flush()

    def close(self, complete=True):
        if self.closed:
            return
        self.finish_episode('writer_closed', valid=False)
        self.policy.attrs['complete'] = bool(complete)
        self.audit.attrs['complete'] = bool(complete)
        self.policy.close()
        self.audit.close()
        self.closed = True
