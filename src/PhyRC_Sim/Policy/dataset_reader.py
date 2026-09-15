"""Lazy, public-only HDF5 reader usable with torch DataLoader or NumPy."""
import json
import os
from pathlib import Path

import numpy as np

from .contract import observation_shapes


class PolicyDataset:
    """Returns only (observation, action); never opens audit or raw archives.

    Each DataLoader process opens its own handle. Camera keys are split by name;
    low-dimensional observations/actions are flattened in row-major robot order.
    Images remain uint8 HWC and metric depth float32 HWC, without preprocessing.
    """
    def __init__(self, path, observation_keys=None):
        import h5py
        self.path = str(Path(path))
        self.file = None
        self.pid = None
        self.index = []
        with h5py.File(self.path, 'r') as f:
            if f.attrs.get('schema') != 'phyrc-policy-dataset-v1' or not f.attrs.get('complete', False):
                raise ValueError('Not a complete PhyRC policy dataset')
            contract = json.loads(f.attrs['contract_json'])
            self.contract = contract
            # Compare to the installed allowlist, not a dataset-supplied list.
            root = Path(__file__).resolve().parents[3]
            if not (root / 'config/policy_interface.json').exists():
                root = Path('/project')
            official = json.loads((root / 'config/policy_interface.json').read_text())
            if contract['observations'] != official['observations'] or contract['action'] != official['action']:
                raise ValueError('Dataset changes the permitted observation/action contract')
            if contract['cameras'] != official['cameras'] or contract['timing'] != official['timing']:
                raise ValueError('Dataset changes camera definitions or timing')
            shapes = observation_shapes(contract, robots=2, joints=13, profile='actor_rgbd')
            self.specs = {}
            for name, shape in shapes.items():
                dtype = np.dtype(contract['observations'][name]['dtype'])
                if name in ('rgb', 'depth', 'depth_valid'):
                    for camera in contract['cameras']:
                        self.specs[camera['id'] + '_' + name] = (shape[1:], dtype)
                else:
                    self.specs[name] = ((int(np.prod(shape)),), dtype)
            self.keys = list(self.specs) if observation_keys is None else list(observation_keys)
            if not self.keys or not set(self.keys) <= set(self.specs):
                raise ValueError('Requested observation is outside the public allowlist')
            for name, demo in f['data'].items():
                if not demo.attrs.get('complete', False):
                    raise ValueError('Incomplete episode: ' + name)
                n = int(demo.attrs['num_samples'])
                for group_name in ('obs', 'next_obs'):
                    if set(demo[group_name]) != set(self.specs):
                        raise ValueError('Unexpected or missing public observation fields')
                    for key, (shape, dtype) in self.specs.items():
                        ds = demo[group_name][key]
                        if ds.shape != (n, *shape) or ds.dtype != dtype:
                            raise ValueError('Observation layout mismatch: ' + key)
                if demo['actions'].shape != (n, 18) or demo['actions'].dtype != np.float32:
                    raise ValueError('Expected normalized float32 (N,18) actions')
                self.index.extend((name, i) for i in range(n))
            if len(self.index) != f['data'].attrs['total'] or not self.index:
                raise ValueError('Empty or inconsistent dataset')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        import h5py
        if self.file is not None and self.pid != os.getpid():
            self.close()
        if self.file is None:
            self.file = h5py.File(self.path, 'r')
            self.pid = os.getpid()
        name, step = self.index[index]
        demo = self.file['data'][name]
        obs = {key: demo['obs'][key][step] for key in self.keys}
        action = demo['actions'][step]
        if not np.isfinite(action).all() or np.any(abs(action) > 1):
            raise ValueError('Invalid action')
        return obs, action

    def __getstate__(self):
        return dict(self.__dict__, file=None)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
