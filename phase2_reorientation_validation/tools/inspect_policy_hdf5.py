#!/usr/bin/env python3
"""Inspect only the allowed `data/demo_N/obs` portion of policy.hdf5."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
from PIL import Image, ImageDraw

from _common import CAMERAS, get_camera_arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, type=Path, help="Path to policy.hdf5")
    parser.add_argument("--export-demo", help="Optional episode name, e.g. demo_0")
    parser.add_argument("--camera", choices=CAMERAS, default="overview")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-image", type=Path, help="Export one RGB frame as PNG")
    args = parser.parse_args()

    if not args.hdf5.is_file():
        parser.error(f"Không tìm thấy HDF5: {args.hdf5}")

    with h5py.File(args.hdf5, "r") as h5:
        if "data" not in h5:
            parser.error("File không có group `data`; đây có thể không phải policy.hdf5.")
        print(f"FILE: {args.hdf5}")
        for demo in sorted(h5["data"].keys()):
            obs_path = f"data/{demo}/obs"
            if obs_path not in h5:
                print(f"\n[{demo}] thiếu obs group")
                continue
            obs = h5[obs_path]
            print(f"\n[{demo}]")
            for camera in CAMERAS:
                for suffix in ("rgb", "depth", "depth_valid"):
                    key = f"{camera}_{suffix}"
                    if key in obs:
                        ds = obs[key]
                        print(f"  {key:24s} shape={str(ds.shape):22s} dtype={ds.dtype}")
                    else:
                        print(f"  {key:24s} MISSING")
            if "simulation_time_s" in obs:
                ts = obs["simulation_time_s"]
                print(f"  {'simulation_time_s':24s} shape={ts.shape} dtype={ts.dtype}")

        if args.export_demo:
            obs = h5[f"data/{args.export_demo}/obs"]
            rgb_ds, _, _ = get_camera_arrays(obs, args.camera)
            if not 0 <= args.frame < rgb_ds.shape[0]:
                parser.error(f"frame phải thuộc [0, {rgb_ds.shape[0]-1}]")
            if args.output_image is None:
                parser.error("Cần --output-image khi dùng --export-demo")
            image = Image.fromarray(rgb_ds[args.frame].astype("uint8"), mode="RGB")
            args.output_image.parent.mkdir(parents=True, exist_ok=True)
            image.save(args.output_image)
            print(f"\nĐã xuất RGB frame: {args.output_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
