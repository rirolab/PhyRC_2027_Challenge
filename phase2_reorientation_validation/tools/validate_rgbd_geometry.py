#!/usr/bin/env python3
"""Offline RGB-D garment-mask diagnostics; public HDF5 observations only.

Streams one 256x256 frame at a time, does not load full HDF5 to RAM or use GPU.
Depth is assumed in metres as declared by the existing camera interface; verify
against official observation specification before interpreting absolute values.
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from _common import CAMERAS, get_camera_arrays, read_depth_frame, read_time
from phase2_reorientation.perception import (
    CameraFrame, MultiHSVRangeSegmenter, PerceptionProcessor,
)


def median_or_none(values):
    return float(np.median(values)) if len(values) else None


def robust_spread(values):
    if len(values) == 0:
        return None
    return float(np.quantile(values, .9) - np.quantile(values, .1))


def depth_diagnostics(mask, depth, valid):
    good = valid & np.isfinite(depth) & (depth > 0)
    within = mask & good
    values = depth[within]
    total = int(mask.sum())

    # Four-neighbour colour-boundary differences where BOTH depth readings are valid.
    # Diagnostic only: an RGB boundary need not be a depth discontinuity.
    deltas = []
    boundary_pairs = 0
    for dy, dx in ((0, 1), (1, 0)):
        if dx:
            ma, mb = mask[:, :-1], mask[:, 1:]
            da, db = depth[:, :-1], depth[:, 1:]
            ga, gb = good[:, :-1], good[:, 1:]
        else:
            ma, mb = mask[:-1, :], mask[1:, :]
            da, db = depth[:-1, :], depth[1:, :]
            ga, gb = good[:-1, :], good[1:, :]
        sel = (ma != mb) & ga & gb
        boundary_pairs += int(sel.sum())
        if sel.any():
            deltas.append(np.abs(da[sel] - db[sel]))

    changes = np.concatenate(deltas) if deltas else np.array([], dtype=np.float32)
    return {
        'mask_depth_valid_fraction': float(len(values) / total) if total else None,
        'depth_median_m': median_or_none(values),
        'depth_p10_p90_span_m': robust_spread(values),
        'depth_p10_m': float(np.quantile(values, .1)) if len(values) else None,
        'depth_p90_m': float(np.quantile(values, .9)) if len(values) else None,
        'boundary_valid_pair_count': boundary_pairs,
        'boundary_median_abs_depth_step_m': median_or_none(changes),
        'boundary_p90_abs_depth_step_m': float(np.quantile(changes, .9)) if len(changes) else None,
        'full_frame_depth_valid_fraction': float(good.mean()),
    }


def save_preview(rgb, depth, valid, detection, row, output):
    # Two-panel PNG: overlay mask and robust depth greyscale. No per-frame arrays stored.
    a = Image.fromarray(rgb, 'RGB').convert('RGBA')
    tint = np.zeros((*detection.mask.shape, 4), dtype=np.uint8)
    tint[detection.mask] = (255, 0, 90, 110)
    a = Image.alpha_composite(a, Image.fromarray(tint, 'RGBA')).convert('RGB')
    if detection.bbox_xyxy is not None:
        d = ImageDraw.Draw(a)
        x1, y1, x2, y2 = detection.bbox_xyxy
        d.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(255, 255, 0), width=2)

    good = valid & np.isfinite(depth) & (depth > 0)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if good.any():
        lo, hi = np.quantile(depth[good], [.02, .98])
        if hi > lo:
            gray[good] = (255 * np.clip((depth[good] - lo) / (hi - lo), 0, 1)).astype(np.uint8)
        else:
            gray[good] = 128
    b = Image.fromarray(gray, 'L').convert('RGB')
    out = Image.new('RGB', (512, 256))
    out.paste(a, (0, 0))
    out.paste(b, (256, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    out.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hdf5', type=Path, required=True)
    p.add_argument('--demo', default='demo_0')
    p.add_argument('--calibration', type=Path, action='append', required=True)
    p.add_argument('--camera', choices=('all',) + CAMERAS, default='all')
    p.add_argument('--every', type=int, default=50)
    p.add_argument('--max-frames', type=int, default=30)
    p.add_argument('--min-component-pixels', type=int, default=80)
    p.add_argument('--keep-components', type=int, default=3)
    p.add_argument('--min-mask-pixels', type=int, default=30)
    p.add_argument('--preview-frames', type=int, nargs='*', default=[0, 600, 1250, 1850, 2600])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.every, a.max_frames, a.min_component_pixels, a.keep_components, a.min_mask_pixels) < 1:
        p.error('Sampling and threshold arguments must be >=1')
    if len(a.calibration) < 2:
        p.error('Provide at least two --calibration values (blue, yellow)')
    ranges = []
    for path in a.calibration:
        c = json.loads(path.read_text(encoding='utf-8'))
        ranges.append((tuple(c['hsv_lower']), tuple(c['hsv_upper'])))
    segmenter = MultiHSVRangeSegmenter(
        hsv_ranges=tuple(ranges),
        min_component_pixels=a.min_component_pixels,
        keep_largest_components=a.keep_components,
    )
    processor = PerceptionProcessor(segmenter=segmenter, min_mask_pixels=a.min_mask_pixels)
    cameras = CAMERAS if a.camera == 'all' else (a.camera,)
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with h5py.File(a.hdf5, 'r') as h5:
        obs = h5[f'data/{a.demo}/obs']
        sources = {cam: get_camera_arrays(obs, cam) for cam in cameras}
        n = min(ds[0].shape[0] for ds in sources.values())
        candidates = list(range(0, n, a.every))
        if len(candidates) > a.max_frames:
            idx = np.linspace(0, len(candidates) - 1, a.max_frames, dtype=int)
            selected = [candidates[i] for i in idx]
        else:
            selected = candidates
        # Add fixed frames to diagnostic inspection but KEEP benchmark sample list unchanged.
        extra = [i for i in a.preview_frames if 0 <= i < n]
        for fi in sorted(set(selected) | set(extra)):
            for cam in cameras:
                rgb_ds, depth_ds, valid_ds = sources[cam]
                rgb = rgb_ds[fi].astype(np.uint8)
                depth = read_depth_frame(depth_ds, fi).astype(np.float32)
                valid = read_depth_frame(valid_ds, fi).astype(bool)
                detection = processor.process_frame(CameraFrame(
                    camera_name=cam, rgb=rgb, depth_m=depth, depth_valid=valid,
                ))
                record = {
                    'camera': cam, 'frame_index': fi, 'time_s': read_time(obs, fi),
                    'is_benchmark_sample': int(fi in selected),
                    'mask_valid': int(detection.valid), 'mask_area_px': int(detection.area_px),
                    'mask_area_fraction': float(detection.area_px / rgb.shape[0] / rgb.shape[1]),
                    **depth_diagnostics(detection.mask, depth, valid),
                }
                rows.append(record)
                if fi in extra:
                    save_preview(rgb, depth, valid, detection, record,
                        a.output / 'previews' / f'{a.demo}_{fi:06d}_{cam}.png')
                print(f"{cam:16s} frame={fi:4d} area={detection.area_px:5d} "
                      f"depth_valid={record['mask_depth_valid_fraction']} "
                      f"span_m={record['depth_p10_p90_span_m']}")
    with (a.output / 'rgbd_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'purpose': 'Descriptive depth diagnostics, NOT segmentation ground truth or 3-D pose',
               'data_source': str(a.hdf5), 'calibrations': list(map(str, a.calibration)),
               'depth_unit_assumption': 'meters; check official dataset documentation',
               'frames_used_for_summary': 'benchmark samples only; extra preview frames excluded',
               'sampling': {'every': a.every, 'max_frames': a.max_frames}, 'cameras': {}}
    for cam in cameras:
        subset = [r for r in rows if r['camera'] == cam and r['is_benchmark_sample']]
        summary['cameras'][cam] = {
            'samples': len(subset),
            'mask_valid_rate': float(np.mean([r['mask_valid'] for r in subset])),
            'median_mask_depth_valid_fraction': median_or_none([
                r['mask_depth_valid_fraction'] for r in subset
                if r['mask_depth_valid_fraction'] is not None]),
            'median_depth_p10_p90_span_m': median_or_none([
                r['depth_p10_p90_span_m'] for r in subset
                if r['depth_p10_p90_span_m'] is not None]),
            'median_boundary_depth_step_m': median_or_none([
                r['boundary_median_abs_depth_step_m'] for r in subset
                if r['boundary_median_abs_depth_step_m'] is not None]),
        }
    summary['linux_maxrss_reported_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (a.output / 'rgbd_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Saved:', a.output / 'rgbd_metrics.csv')
    print('Saved:', a.output / 'rgbd_summary.json')
    print('Peak process RSS (Linux):', summary['linux_maxrss_reported_kib'] / 1024, 'MiB')


if __name__ == '__main__':
    main()
