"""Shared helpers for offline PhyRC policy.hdf5 perception validation."""
from __future__ import annotations

import sys
from pathlib import Path

# Make `phase2_reorientation` importable when a tool is run by file path
# from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CAMERAS = (
    "overview",
    "robot_0_wrist",
    "robot_1_wrist",
    "robot_0_head",
    "robot_1_head",
)


def get_episode(h5, demo: str):
    path = f"data/{demo}/obs"
    if path not in h5:
        demos = sorted(h5.get("data", {}).keys()) if "data" in h5 else []
        raise KeyError(f"Không thấy group {path}. Episodes hiện có: {demos}")
    return h5[path]


def get_camera_arrays(obs_group, camera: str):
    names = (f"{camera}_rgb", f"{camera}_depth", f"{camera}_depth_valid")
    missing = [name for name in names if name not in obs_group]
    if missing:
        raise KeyError(f"Thiếu dataset trong obs: {missing}")
    rgb_ds, depth_ds, valid_ds = (obs_group[name] for name in names)
    if len(rgb_ds.shape) != 4 or rgb_ds.shape[1:] != (256, 256, 3):
        raise ValueError(f"{names[0]} phải có shape (N,256,256,3), nhận {rgb_ds.shape}")
    if len(depth_ds.shape) not in (3, 4):
        raise ValueError(f"{names[1]} có shape không hỗ trợ: {depth_ds.shape}")
    if len(valid_ds.shape) not in (3, 4):
        raise ValueError(f"{names[2]} có shape không hỗ trợ: {valid_ds.shape}")
    if depth_ds.shape[0] != rgb_ds.shape[0] or valid_ds.shape[0] != rgb_ds.shape[0]:
        raise ValueError(f"Số frame RGB/depth/depth_valid không khớp cho {camera}")
    if depth_ds.shape[1:3] != (256, 256) or valid_ds.shape[1:3] != (256, 256):
        raise ValueError(f"Kích thước depth/depth_valid không phải 256x256 cho {camera}")
    if depth_ds.ndim == 4 and depth_ds.shape[-1] != 1:
        raise ValueError(f"{names[1]} cần channel cuối = 1, nhận {depth_ds.shape}")
    if valid_ds.ndim == 4 and valid_ds.shape[-1] != 1:
        raise ValueError(f"{names[2]} cần channel cuối = 1, nhận {valid_ds.shape}")
    return rgb_ds, depth_ds, valid_ds


def read_depth_frame(dataset, index):
    value = dataset[index]
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    return value


def read_time(obs_group, index: int) -> float:
    if "simulation_time_s" in obs_group:
        value = obs_group["simulation_time_s"][index]
        return float(value.reshape(-1)[0] if getattr(value, "ndim", 0) else value)
    # Fallback is explicitly an approximation based on the documented 20 Hz.
    return float(index) * 0.05
