"""Read preserved teleoperation stages with current physics settings."""
from pathlib import Path
import numpy as np


def validate_slot(path, revision):
    path = Path(path)
    if not path.name.startswith('slot_') or path.suffix != '.npz':
        raise ValueError('Expected slot_NAME.npz')
    with np.load(path, allow_pickle=False) as data:
        if str(data['scene_geometry_revision']) != revision:
            raise ValueError('Stage geometry differs from current simulator')
        for key in ('g0_pos', 'g0_vel', 'placement_human', 'placement_chair'):
            if not np.isfinite(data[key]).all():
                raise ValueError(f'Nonfinite stage field: {key}')


def restore_placement(stage, path):
    from pxr import Gf, Sdf, Usd, UsdGeom
    from Env_Config.Human.RandomSpawn import placement_matches
    with np.load(path, allow_pickle=False) as data:
        UsdGeom.Xform.Define(stage, '/World/Chair')
        for name in ('Human', 'Chair'):
            prim = stage.GetPrimAtPath('/World/' + name)
            desired = Gf.Matrix4d(data['placement_' + name.lower()].tolist())
            if name == 'Human':
                original = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                # MakeMatrixXform folds away the named randomSpawn op. Retain
                # its equivalent delta as camera-only metadata, not an xform.
                prim.CreateAttribute('phyrc:cameraSpawnDelta', Sdf.ValueTypeNames.Matrix4d,
                                     custom=True).Set(original.GetInverse() * desired)
            parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(desired * parent.GetInverse())
        if not placement_matches(stage, data):
            raise RuntimeError('Failed to restore static placement before physics initialization')
    return {'restored_from': str(path), 'randomized': False}


def load_slot(env, path):
    path = Path(path)
    validate_slot(path, env.M._STATE_GEOMETRY_REVISION)
    with np.load(path, allow_pickle=False) as data:
        if int(data['n_garments']) != len(env.cloths) or int(data['n_rigs']) != len(env.rigs):
            raise ValueError('Stage object counts differ from the live scene')
        for i, cloth in enumerate(env.cloths):
            shape = tuple(cloth.get_world_positions().shape[1:])
            for suffix in ('pos', 'vel'):
                values = data[f'g{i}_{suffix}']
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError('Invalid cloth state; refusing partial restore')
        for i, rig in enumerate(env.rigs):
            if data[f'r{i}_jpos'].shape != tuple(rig['robot'].get_joint_positions().shape):
                raise ValueError('Joint layout differs; refusing partial restore')
    # Production loader clears native grasps, contact caches and command history.
    previous = env.M.STATE_DIR
    try:
        env.M.STATE_DIR = str(path.parent)
        if not env.M.load_state_slot(path.stem.removeprefix('slot_'), env.cloths, env.rigs):
            raise RuntimeError(f'Failed to load complete stage: {path}')
        env._slot_load_count = getattr(env, '_slot_load_count', 0) + 1
    finally:
        env.M.STATE_DIR = previous
