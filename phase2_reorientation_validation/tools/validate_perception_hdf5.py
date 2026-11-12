#!/usr/bin/env python3
"""Run the current PerceptionProcessor over allowed RGB-D from policy.hdf5."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from _common import CAMERAS, get_camera_arrays, read_depth_frame, read_time
from phase2_reorientation.perception import (
    CameraFrame,
    HSVRangeSegmenter,
    MultiHSVRangeSegmenter,
    PerceptionProcessor,
)


def parse_bounds(text: str, name: str):
    try:
        values = tuple(float(v.strip()) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} phải có dạng H,S,V") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{name} phải gồm 3 số H,S,V")
    return values


def render_camera(rgb, detection, camera, demo, frame_idx, sim_time, out_path):
    base = Image.fromarray(rgb.astype(np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    pixels = np.zeros((256, 256, 4), dtype=np.uint8)
    pixels[detection.mask] = (255, 40, 40, 100)
    overlay = Image.fromarray(pixels, mode="RGBA")
    preview = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(preview)
    if detection.bbox_xyxy is not None:
        x1, y1, x2, y2 = detection.bbox_xyxy
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(255, 255, 0), width=2)
    if detection.centroid_xy is not None:
        cx, cy = detection.centroid_xy.tolist()
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 255, 0))
    if detection.valid and detection.principal_axis_xy is not None:
        axis = detection.principal_axis_xy
        cx, cy = detection.centroid_xy.tolist()
        length = 55.0
        dx, dy = float(axis[0]) * length, float(axis[1]) * length
        draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=(0, 220, 255), width=3)
    status = "VALID" if detection.valid else f"INVALID: {detection.reason}"
    draw.rectangle((0, 0, 255, 34), fill=(0, 0, 0))
    draw.text((5, 5), f"{camera} | {demo} | frame={frame_idx} t={sim_time:.2f}s", fill="white")
    draw.text((5, 19), f"{status} area={detection.area_px} conf={detection.geometry_confidence:.3f}", fill="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--demo", required=True, help="Episode, e.g. demo_0")
    parser.add_argument("--calibration", required=True, type=Path, action="append",
                        help="HSV calibration JSON; repeat for multi-colour segmentation")
    parser.add_argument("--camera", choices=("all",) + CAMERAS, default="all")
    parser.add_argument("--every", type=int, default=10, help="Sample every N steps; default 10 = 2 Hz")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum sampled frames, default 60")
    parser.add_argument("--min-mask-pixels", type=int, default=30)
    parser.add_argument("--min-component-pixels", type=int, default=80)
    parser.add_argument("--keep-components", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("research_output/perception_validation"))
    args = parser.parse_args()

    if args.every < 1 or args.max_frames < 1:
        parser.error("--every và --max-frames phải >= 1")
    if not args.hdf5.is_file() or any(not p.is_file() for p in args.calibration):
        parser.error("Không tìm thấy policy.hdf5 hoặc file calibration JSON")
    if args.min_component_pixels < 1 or args.keep_components < 1:
        parser.error("--min-component-pixels và --keep-components phải >= 1")

    calibrations = [json.loads(p.read_text(encoding="utf-8")) for p in args.calibration]
    hsv_ranges = [(tuple(c["hsv_lower"]), tuple(c["hsv_upper"])) for c in calibrations]
    if len(hsv_ranges) == 1:
        lower, upper = hsv_ranges[0]
        segmenter = HSVRangeSegmenter(hsv_lower=lower, hsv_upper=upper)
        segmenter_name = "single_hsv"
    else:
        segmenter = MultiHSVRangeSegmenter(
            hsv_ranges=tuple(hsv_ranges),
            min_component_pixels=args.min_component_pixels,
            keep_largest_components=args.keep_components,
        )
        segmenter_name = "multi_hsv"
    processor = PerceptionProcessor(segmenter=segmenter, min_mask_pixels=args.min_mask_pixels)
    cameras = CAMERAS if args.camera == "all" else (args.camera,)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "metrics.csv"
    rows = []
    stats = defaultdict(lambda: {"sampled": 0, "valid": 0, "reasons": Counter(),
                                 "areas": [], "confidences": [], "depth_fractions": [],
                                 "centroid_steps": [], "axis_steps": [], "temporal_pairs": 0})
    previous = {camera: None for camera in cameras}

    with h5py.File(args.hdf5, "r") as h5:
        obs = h5[f"data/{args.demo}/obs"]
        camera_data = {cam: get_camera_arrays(obs, cam) for cam in cameras}
        lengths = {cam: arrs[0].shape[0] for cam, arrs in camera_data.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Frame counts across cameras differ: {lengths}")
        total_frames = next(iter(lengths.values()))
        candidates = list(range(0, total_frames, args.every))
        if len(candidates) > args.max_frames:
            pick = np.linspace(0, len(candidates) - 1, num=args.max_frames, dtype=int)
            selected = [candidates[i] for i in pick]
        else:
            selected = candidates
        print(f"Episode={args.demo}; frames={total_frames}; sampled={len(selected)}; cameras={cameras}")

        for frame_idx in selected:
            sim_time = read_time(obs, frame_idx)
            for camera in cameras:
                rgb_ds, depth_ds, valid_ds = camera_data[camera]
                rgb = rgb_ds[frame_idx].astype(np.uint8)
                depth = read_depth_frame(depth_ds, frame_idx).astype(np.float32)
                depth_valid = read_depth_frame(valid_ds, frame_idx).astype(bool)
                frame = CameraFrame(camera_name=camera, rgb=rgb, depth_m=depth,
                                    depth_valid=depth_valid)
                detection = processor.process_frame(frame)
                area_fraction = detection.area_px / float(256 * 256)
                angle = detection.angle_image_rad
                centroid_step = ""
                axis_step = ""
                if detection.valid and detection.centroid_xy is not None and angle is not None:
                    prev = previous[camera]
                    if prev is not None:
                        centroid_step = float(np.linalg.norm(detection.centroid_xy - prev["centroid"]))
                        delta = abs(float(angle) - prev["angle"]) % np.pi
                        axis_step = float(np.degrees(min(delta, np.pi - delta)))
                        stats[camera]["centroid_steps"].append(centroid_step)
                        stats[camera]["axis_steps"].append(axis_step)
                        stats[camera]["temporal_pairs"] += 1
                    previous[camera] = {"centroid": detection.centroid_xy.copy(), "angle": float(angle)}
                else:
                    previous[camera] = None

                row = {
                    "demo": args.demo,
                    "frame_index": frame_idx,
                    "simulation_time_s": sim_time,
                    "camera": camera,
                    "valid": int(detection.valid),
                    "reason": detection.reason,
                    "area_px": detection.area_px,
                    "area_fraction": area_fraction,
                    "bbox_x1": detection.bbox_xyxy[0] if detection.bbox_xyxy else "",
                    "bbox_y1": detection.bbox_xyxy[1] if detection.bbox_xyxy else "",
                    "bbox_x2": detection.bbox_xyxy[2] if detection.bbox_xyxy else "",
                    "bbox_y2": detection.bbox_xyxy[3] if detection.bbox_xyxy else "",
                    "centroid_x": float(detection.centroid_xy[0]) if detection.centroid_xy is not None else "",
                    "centroid_y": float(detection.centroid_xy[1]) if detection.centroid_xy is not None else "",
                    "principal_axis_angle_rad_mod_pi": angle if angle is not None else "",
                    "median_depth_m": detection.median_depth_m if detection.median_depth_m is not None else "",
                    "depth_valid_fraction": detection.depth_valid_fraction,
                    "geometry_confidence": detection.geometry_confidence,
                    "centroid_step_from_previous_sample_px": centroid_step,
                    "axis_step_from_previous_sample_deg": axis_step,
                }
                rows.append(row)
                s = stats[camera]
                s["sampled"] += 1
                s["valid"] += int(detection.valid)
                if not detection.valid:
                    s["reasons"][detection.reason] += 1
                s["areas"].append(area_fraction)
                s["confidences"].append(detection.geometry_confidence)
                s["depth_fractions"].append(detection.depth_valid_fraction)

                image_path = args.output / "overlays" / f"{args.demo}_{frame_idx:06d}_{camera}.png"
                render_camera(rgb, detection, camera, args.demo, frame_idx, sim_time, image_path)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "dataset": str(args.hdf5),
        "episode": args.demo,
        "segmenter": segmenter_name,
        "calibrations": [str(p) for p in args.calibration],
        "hsv_ranges": hsv_ranges,
        "min_component_pixels": args.min_component_pixels if len(hsv_ranges) > 1 else None,
        "keep_largest_components": args.keep_components if len(hsv_ranges) > 1 else None,
        "sample_every_n_steps": args.every,
        "sampled_frames_per_camera": len(selected),
        "cameras": {},
        "important_limit": "Geometry confidence is heuristic; it is not semantic garment-classification accuracy.",
    }
    for camera in cameras:
        s = stats[camera]
        n = s["sampled"]
        summary["cameras"][camera] = {
            "sampled": n,
            "valid": s["valid"],
            "valid_rate": (s["valid"] / n) if n else 0.0,
            "median_area_fraction": float(np.median(s["areas"])) if n else 0.0,
            "median_geometry_confidence": float(np.median(s["confidences"])) if n else 0.0,
            "median_depth_valid_fraction": float(np.median(s["depth_fractions"])) if n else 0.0,
            "temporal_pair_count": s["temporal_pairs"],
            "median_centroid_step_px": float(np.median(s["centroid_steps"])) if s["centroid_steps"] else None,
            "p90_centroid_step_px": float(np.quantile(s["centroid_steps"], 0.90)) if s["centroid_steps"] else None,
            "median_axis_step_deg": float(np.median(s["axis_steps"])) if s["axis_steps"] else None,
            "p90_axis_step_deg": float(np.quantile(s["axis_steps"], 0.90)) if s["axis_steps"] else None,
            "invalid_reasons": dict(s["reasons"]),
        }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved metrics: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved overlays: {args.output / 'overlays'}")
    for camera, data in summary["cameras"].items():
        centroid = data["median_centroid_step_px"]
        axis = data["median_axis_step_deg"]
        print(f"{camera:16s} valid_rate={data['valid_rate']:.1%} "
              f"median_area={data['median_area_fraction']:.3f} "
              f"median_conf={data['median_geometry_confidence']:.3f} "
              f"centroid_step={centroid if centroid is not None else 'N/A'} "
              f"axis_step_deg={axis if axis is not None else 'N/A'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
