#!/usr/bin/env python3
"""Estimate HSV bounds from a manually selected garment-only ROI in policy.hdf5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from _common import CAMERAS, get_camera_arrays
from phase2_reorientation.perception import _rgb_to_hsv


def circular_hue_interval(hues: np.ndarray, q_low: float, q_high: float):
    values = np.sort(np.mod(hues.astype(np.float64), 360.0))
    if values.size == 0:
        raise ValueError("ROI không có pixel để hiệu chỉnh")
    if values.size == 1:
        return float(values[0]), float(values[0])
    gaps = np.diff(np.concatenate([values, values[:1] + 360.0]))
    cut = int(np.argmax(gaps))
    start = (cut + 1) % values.size
    rotated = np.concatenate([values[start:], values[:start] + 360.0])
    lo = float(np.quantile(rotated, q_low) % 360.0)
    hi = float(np.quantile(rotated, q_high) % 360.0)
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--demo", required=True, help="Episode, e.g. demo_0")
    parser.add_argument("--camera", required=True, choices=CAMERAS)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Rectangle pixels, x1 y1 x2 y2; crop ONLY garment fabric")
    parser.add_argument("--output", type=Path, default=Path("hsv_calibration.json"))
    parser.add_argument("--preview", type=Path, default=Path("hsv_roi_preview.png"))
    args = parser.parse_args()

    x1, y1, x2, y2 = args.roi
    if not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256):
        parser.error("ROI phải thỏa 0 <= x1 < x2 <= 256 và 0 <= y1 < y2 <= 256")

    with h5py.File(args.hdf5, "r") as h5:
        obs = h5[f"data/{args.demo}/obs"]
        rgb_ds, _, _ = get_camera_arrays(obs, args.camera)
        if not 0 <= args.frame < rgb_ds.shape[0]:
            parser.error(f"frame phải thuộc [0, {rgb_ds.shape[0]-1}]")
        rgb = rgb_ds[args.frame].astype(np.uint8)

    roi_rgb = rgb[y1:y2, x1:x2]
    hsv = _rgb_to_hsv(roi_rgb).reshape(-1, 3)

    # Robustly estimate the dominant colour population in the manually
    # selected ROI. A small number of boundary/reflection pixels must not
    # expand the Hue interval across unrelated colours.
    hues = np.mod(hsv[:, 0].astype(np.float64), 360.0)

    bin_width_deg = 10.0
    edges = np.arange(0.0, 360.0 + bin_width_deg, bin_width_deg)
    hist, _ = np.histogram(hues, bins=edges)

    dominant_bin = int(np.argmax(hist))
    dominant_center = (
        edges[dominant_bin] + edges[dominant_bin + 1]
    ) / 2.0

    # Circular angular distance from dominant Hue.
    hue_distance = np.abs(
        ((hues - dominant_center + 180.0) % 360.0) - 180.0
    )

    # Manual ROI is expected to contain one garment colour.
    # Keep a broad +/- 35 degree neighbourhood around its dominant Hue.
    dominant = hue_distance <= 35.0

    if np.count_nonzero(dominant) < max(30, int(0.50 * hsv.shape[0])):
        raise ValueError(
            "ROI does not contain a sufficiently dominant single colour"
        )

    hsv_used = hsv[dominant]

    h_lo, h_hi = circular_hue_interval(
        hsv_used[:, 0], 0.02, 0.98
    )
    s_lo, s_hi = np.quantile(
        hsv_used[:, 1], [0.02, 0.98]
    )
    v_lo, v_hi = np.quantile(
        hsv_used[:, 2], [0.02, 0.98]
    )
    lower = [h_lo, max(0.0, float(s_lo) - 0.03), max(0.0, float(v_lo) - 0.03)]
    upper = [h_hi, min(1.0, float(s_hi) + 0.03), min(1.0, float(v_hi) + 0.03)]

    result = {
        "source": "manual_garment_roi_from_policy_hdf5",
        "demo": args.demo,
        "camera": args.camera,
        "frame": args.frame,
        "roi_xyxy": [x1, y1, x2, y2],
        "sample_pixels": int(hsv.shape[0]),
        "hsv_lower": lower,
        "hsv_upper": upper,
        "note": "Initial measured HSV interval only; inspect false positives and tune on held-out frames.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    preview = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(preview)
    draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(255, 255, 0), width=2)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    preview.save(args.preview)

    print(json.dumps(result, indent=2))
    print(f"Saved calibration: {args.output}")
    print(f"Saved ROI preview: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
