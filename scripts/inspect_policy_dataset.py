#!/usr/bin/env python3
"""CPU dataset inspection and public observation/action contract validation."""
import argparse
import json
from pathlib import Path
import sys

root = Path('/project') if Path('/project/config').exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src/PhyRC_Sim'))
from Policy.dataset_reader import PolicyDataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset', type=Path, help='Path to policy.hdf5 (not audit.hdf5)')
args = parser.parse_args()
dataset = PolicyDataset(args.dataset)
try:
    obs, action = dataset[0]
    print(json.dumps({'samples': len(dataset), 'action_shape': list(action.shape),
                      'observations': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in obs.items()},
                      'note': 'Schema inspection, not a sandbox or proof of training-data compliance'}, indent=2))
finally:
    dataset.close()
