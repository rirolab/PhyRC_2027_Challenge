"""Uniform choice of a support box, independent of the human RNG stream."""
import os
import numpy as np
from Env_Config.Randomization import spawn_randomization_enabled

# Domain separation keeps adding garment randomness from changing human poses.
_GARMENT_STREAM = 0x53484952


def sample_garment_spawn(table_centers, table_size, *, seed=None, clearance=0.2):
    centers = np.asarray(table_centers, dtype=np.float64)
    size = np.asarray(table_size, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError('Expected at least one table center [x,y,z]')
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Table size must be three finite positive lengths')
    if not np.isfinite(centers).all() or not np.isfinite(clearance) or clearance < 0:
        raise ValueError('Invalid table coordinates or spawn clearance')
    randomized = spawn_randomization_enabled()
    if randomized:
        if seed is None:
            seed = os.environ.get('STRETCH4_GARMENT_SPAWN_SEED', os.environ.get('HUMAN_SPAWN_SEED'))
            seed = np.random.SeedSequence().entropy if seed is None else int(seed)
        seed = int(seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed, _GARMENT_STREAM]))
        index = int(rng.integers(len(centers)))
    else:
        # Blue was the third color/table pair before randomization was added.
        if len(centers) < 3:
            raise ValueError('Original blue shirt placement requires the third table')
        index, seed = 2, None
    position = centers[index] + [0, 0, size[2] / 2 + clearance]
    return {'seed': seed, 'randomized': randomized, 'table_index': index, 'table_count': len(centers),
            'table_prim_path': f'/World/garment_table_{index}',
            'table_center_world_m': centers[index].tolist(), 'table_size_m': size.tolist(),
            'spawn_position_world_m': position.tolist(), 'clearance_m': float(clearance)}
