#!/usr/bin/env python3
"""Calibrate one blue HSV interval from nine hand-reviewed RGB ROIs in policy.hdf5.

Run from the PhyRC repository root with the project virtual environment active.
This script does not modify the HDF5 file or existing calibrations.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from phase2_reorientation.perception import _rgb_to_hsv

# (camera, frame, x1, y1, x2, y2); manually reviewed on source RGB frames.
ROIS = [
    ("robot_1_head", 0, 105, 125, 125, 145),
    ("robot_1_head", 0, 135, 125, 155, 145),
    ("robot_1_head", 600, 135, 150, 155, 170),
    ("robot_1_head", 600, 190, 155, 210, 175),
    ("robot_1_head", 600, 160, 170, 180, 190),
    ("robot_1_wrist", 1250, 0, 100, 20, 120),
    ("robot_1_wrist", 1250, 30, 110, 50, 130),
    ("robot_1_wrist", 1250, 60, 110, 80, 130),
    ("robot_1_wrist", 1250, 0, 130, 20, 150),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.hdf5.is_file():
        parser.error(f"HDF5 not found: {args.hdf5}")

    selected = []
    reports = []
    with h5py.File(args.hdf5, "r") as h5:
        obs = h5["data/demo_0/obs"]
        for camera, frame, x1, y1, x2, y2 in ROIS:
            ds = obs[f"{camera}_rgb"]
            if not (0 <= frame < ds.shape[0] and 0 <= x1 < x2 <= ds.shape[2]
                    and 0 <= y1 < y2 <= ds.shape[1]):
                raise ValueError(f"Out-of-bounds ROI: {camera} {frame} {(x1,y1,x2,y2)}")
            rgb = ds[frame, y1:y2, x1:x2, :]
            hsv = _rgb_to_hsv(rgb).reshape(-1, 3)
            # Circular Hue distance to blue (240 degrees), rejecting non-blue ROI contamination.
            hue_dist = np.abs((hsv[:, 0] - 240.0 + 180.0) % 360.0 - 180.0)
            keep = (hue_dist <= 25.0) & (hsv[:, 1] >= 0.25)
            purity = float(keep.mean())
            if purity < 0.85:
                raise ValueError(f"ROI is not sufficiently blue: {camera}@{frame}, purity={purity:.1%}")
            selected.append(hsv[keep])
            reports.append({"camera": camera, "frame": frame,
                            "roi_xyxy": [x1, y1, x2, y2],
                            "blue_pixel_fraction": round(purity, 4),
                            "selected_pixels": int(keep.sum())})
            print(f"{camera:15s} frame={frame:4d} roi={(x1,y1,x2,y2)} blue={purity:.1%}")

    data = np.concatenate(selected, axis=0)
    # Quantiles in an unwrapped Hue coordinate centered at 240 degrees.
    hue_unwrapped = 240.0 + ((data[:, 0] - 240.0 + 180.0) % 360.0 - 180.0)
    hue_lo, hue_hi = np.quantile(hue_unwrapped, [0.01, 0.99])
    sat_lo, sat_hi = np.quantile(data[:, 1], [0.01, 0.99])
    val_lo, val_hi = np.quantile(data[:, 2], [0.01, 0.99])
    # Small, explicit margins to cover quantization and near-boundary illumination.
    lower = [float((hue_lo - 2.0) % 360), float(max(0, sat_lo - 0.05)),
             float(max(0, val_lo - 0.05))]
    upper = [float((hue_hi + 2.0) % 360), float(min(1, sat_hi + 0.05)),
             float(min(1, val_hi + 0.05))]
    result = {
        "source": "multi_roi_blue_from_policy_hdf5",
        "demo": "demo_0",
        "sample_pixels": int(data.shape[0]),
        "hsv_lower": lower,
        "hsv_upper": upper,
        "roi_reports": reports,
        "method": "Blue-pixel filtering; pooled 1st–99th quantiles with H±2°, S/V±0.05 margins",
        "note": "Candidate blue HSV calibration only; validate against held-out frames and overlays.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nBlue HSV lower:", lower)
    print("Blue HSV upper:", upper)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
