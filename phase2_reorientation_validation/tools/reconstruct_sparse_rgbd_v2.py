#!/usr/bin/env python3
"""PhyRC Sparse RGB-D v2: optical-frame validation and sampler A/B benchmark.

Reads only the documented policy.hdf5 observations and static camera HFOV.
Keeps the legacy v1 sampler as control and compares an 8x8 stratified sampler.
No audit, privileged state, camera-to-world transform or world-frame fusion.

INSTALL alongside _common.py and reconstruct_sparse_rgbd.py (v1).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import resource
from collections import deque
from pathlib import Path

import h5py
import numpy as np

from _common import CAMERAS, get_camera_arrays, read_depth_frame, read_time
from phase2_reorientation.perception import CameraFrame, MultiHSVRangeSegmenter, PerceptionProcessor
from reconstruct_sparse_rgbd import DEFAULT_HFOV, camera_matrix, sparse_points, geometry, preview

CLIP = {'overview': 10.0, 'robot_0_head': 5.0, 'robot_1_head': 5.0,
        'robot_0_wrist': 3.0, 'robot_1_wrist': 3.0}


def stratified_flat(good: np.ndarray, max_points: int, grid: int) -> np.ndarray:
    """Deterministic, interleaved, grid-stratified subsample of valid pixels.

    All non-empty cells get a point before second passes when budget permits.
    This is a sampling strategy, not semantic cloth segmentation.
    """
    h, w = good.shape
    flat = np.flatnonzero(good)
    if len(flat) <= max_points:
        return flat
    vv, uu = np.divmod(flat, w)
    cell_x = np.minimum((uu * grid) // w, grid - 1)
    cell_y = np.minimum((vv * grid) // h, grid - 1)
    bucket_id = cell_y * grid + cell_x
    buckets = [flat[bucket_id == k] for k in range(grid * grid)]
    active = [b for b in buckets if len(b)]
    if not active:
        return np.empty((0,), dtype=np.int64)
    # Fair round-robin with deterministic spread *within* each cell.
    # Index over each bucket's local array is staggered over budget rounds.
    counts = np.zeros(len(active), dtype=np.int64)
    for i in range(max_points):
        # Available buckets only: never return repeated indices.
        j = i % len(active)
        if counts[j] >= len(active[j]):
            found = False
            for shift in range(1, len(active) + 1):
                j2 = (j + shift) % len(active)
                if counts[j2] < len(active[j2]):
                    j = j2
                    found = True
                    break
            if not found:
                break
        counts[j] += 1
    chosen = []
    for b, count in zip(active, counts):
        if count:
            loc = (np.arange(count, dtype=np.int64) * len(b)) // count
            chosen.append(b[loc])
    return np.sort(np.concatenate(chosen))


def reproject_indices(flat: np.ndarray, depth: np.ndarray, K: np.ndarray):
    if not len(flat):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)
    vv, uu = np.divmod(flat, depth.shape[1])
    z = depth[vv, uu].astype(np.float64)
    xyz = np.column_stack(((uu-K[0, 2])*z/K[0, 0], (vv-K[1, 2])*z/K[1, 1], z)).astype(np.float32)
    return xyz, np.column_stack((uu, vv)).astype(np.int32)


def components8(mask: np.ndarray):
    """Connected components on post-HSV mask; at most 256x256 pixels."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=np.bool_)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        if seen[y, x]:
            continue
        q = deque([(int(y), int(x))]); seen[y, x] = True
        count = 0
        while q:
            cy, cx = q.popleft(); count += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        sizes.append(count)
    return sorted(sizes, reverse=True)


def occupied_cells(mask: np.ndarray, grid: int) -> int:
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0
    return int(np.unique((ys * grid // h) * grid + xs * grid // w).size)


def sample_metric(xyz: np.ndarray, ref: np.ndarray):
    """Compare sampled centroid and covariance to *all masked depth points*.

    Sampling fidelity only; does NOT validate HSV/3D against ground truth.
    """
    if len(xyz) < 3 or len(ref) < 3:
        return None, None
    sm = xyz.astype(np.float64); rm = ref.astype(np.float64)
    centroid_error = float(np.linalg.norm(sm.mean(axis=0) - rm.mean(axis=0)))
    cov_error = float(np.linalg.norm(np.cov(sm, rowvar=False) - np.cov(rm, rowvar=False), ord='fro'))
    return centroid_error, cov_error


def border_fraction(mask: np.ndarray, width: int = 4) -> float:
    area = int(mask.sum())
    if area == 0:
        return 0.0
    border = np.zeros_like(mask, dtype=bool)
    border[:width, :] = True; border[-width:, :] = True
    border[:, :width] = True; border[:, -width:] = True
    return float((mask & border).sum() / area)


def masked_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def num_or_none(v):
    return None if v is None or not np.isfinite(float(v)) else float(v)


def median(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
    return float(np.median(v)) if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hdf5', type=Path, required=True)
    ap.add_argument('--demo', default='demo_0')
    ap.add_argument('--calibration', type=Path, action='append', required=True)
    ap.add_argument('--camera', choices=('all',) + CAMERAS, default='all')
    ap.add_argument('--every', type=int, default=50)
    ap.add_argument('--max-frames', type=int, default=30)
    ap.add_argument('--max-points', type=int, default=2048)
    ap.add_argument('--strata-grid', type=int, default=8)
    ap.add_argument('--coverage-grid', type=int, default=16)
    ap.add_argument('--min-component-pixels', type=int, default=80)
    ap.add_argument('--keep-components', type=int, default=3)
    ap.add_argument('--min-mask-pixels', type=int, default=30)
    ap.add_argument('--preview-frames', type=int, nargs='*', default=[600, 1250, 2600])
    ap.add_argument('--save-preview-points', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    for key in ('every', 'max_frames', 'max_points', 'strata_grid', 'coverage_grid',
                'min_component_pixels', 'keep_components', 'min_mask_pixels'):
        if getattr(a, key) < 1:
            ap.error(f'--{key.replace("_", "-")} must be >=1')
    if len(a.calibration) < 2:
        ap.error('Both blue-v2 and yellow-v3 --calibration JSONs are required')
    if not a.hdf5.is_file() or any(not p.is_file() for p in a.calibration):
        ap.error('Missing HDF5 or HSV calibration JSON')
    ranges = []
    for p in a.calibration:
        spec = json.loads(p.read_text(encoding='utf-8'))
        ranges.append((tuple(spec['hsv_lower']), tuple(spec['hsv_upper'])))
    segmenter = MultiHSVRangeSegmenter(hsv_ranges=tuple(ranges),
        min_component_pixels=a.min_component_pixels, keep_largest_components=a.keep_components)
    processor = PerceptionProcessor(segmenter=segmenter, min_mask_pixels=a.min_mask_pixels)
    cameras = CAMERAS if a.camera == 'all' else (a.camera,)
    rows = []
    a.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.hdf5, 'r') as h5:
        obs = h5[f'data/{a.demo}/obs']
        ds = {c: get_camera_arrays(obs, c) for c in cameras}
        n = min(v[0].shape[0] for v in ds.values())
        candidates = list(range(0, n, a.every))
        if len(candidates) > a.max_frames:
            selected = {candidates[i] for i in np.linspace(0, len(candidates)-1, a.max_frames, dtype=int)}
        else:
            selected = set(candidates)
        extras = {i for i in a.preview_frames if 0 <= i < n}
        for fi in sorted(selected | extras):
            for cam in cameras:
                rgb_ds, d_ds, valid_ds = ds[cam]
                rgb = rgb_ds[fi].astype(np.uint8)
                depth = read_depth_frame(d_ds, fi).astype(np.float32)
                valid = read_depth_frame(valid_ds, fi).astype(bool)
                h, w = rgb.shape[:2]
                K = camera_matrix(w, h, DEFAULT_HFOV[cam])
                detection = processor.process_frame(CameraFrame(camera_name=cam,
                    rgb=rgb, depth_m=depth, depth_valid=valid))
                mask = np.asarray(detection.mask, dtype=bool)
                depth_ok = valid & np.isfinite(depth) & (depth > 0) & (depth < CLIP[cam])
                good = mask & depth_ok
                n_avail = int(good.sum())
                # Exactly the legacy v1 control for the A/B test.
                legacy_xyz, legacy_uv, _ = sparse_points(mask, depth, valid, K, a.max_points, 0., CLIP[cam])
                flat_stratified = stratified_flat(good, a.max_points, a.strata_grid)
                strat_xyz, strat_uv = reproject_indices(flat_stratified, depth, K)
                full_flat = np.flatnonzero(good)
                # Reference is only used *per frame*, never archived or merged across cameras.
                reference_xyz, _ = reproject_indices(full_flat, depth, K)
                legacy_mean_err, legacy_cov_err = sample_metric(legacy_xyz, reference_xyz)
                strat_mean_err, strat_cov_err = sample_metric(strat_xyz, reference_xyz)
                legacy_grid_mask = np.zeros_like(mask); legacy_grid_mask[legacy_uv[:, 1], legacy_uv[:, 0]] = True
                strat_grid_mask = np.zeros_like(mask); strat_grid_mask[strat_uv[:, 1], strat_uv[:, 0]] = True
                available_cells = occupied_cells(good, a.coverage_grid)
                old_cells = occupied_cells(legacy_grid_mask, a.coverage_grid)
                new_cells = occupied_cells(strat_grid_mask, a.coverage_grid)
                g = geometry(strat_xyz, strat_uv, K)
                components = components8(mask)
                bbox = masked_bbox(mask)
                edge = border_fraction(mask)
                # No universal numeric confidence score: higher area often means background spill.
                row = {'camera': cam, 'frame_index': int(fi), 'time_s': read_time(obs, fi),
                    'is_benchmark_sample': int(fi in selected), 'mask_valid': int(detection.valid),
                    'mask_area_px': int(mask.sum()), 'mask_area_fraction': float(mask.mean()),
                    'mask_components': len(components), 'largest_component_fraction':
                    float(components[0]/sum(components)) if components else None,
                    'mask_border_fraction_4px': edge,
                    'mask_touches_image_edge': int(bool(bbox and (bbox[0] == 0 or bbox[1] == 0 or bbox[2] == w-1 or bbox[3] == h-1))),
                    'bbox_x_min': bbox[0] if bbox else None, 'bbox_y_min': bbox[1] if bbox else None,
                    'bbox_x_max': bbox[2] if bbox else None, 'bbox_y_max': bbox[3] if bbox else None,
                    'mask_depth_valid_fraction': float(n_avail / mask.sum()) if mask.any() else None,
                    'available_mask_depth_points': n_avail, 'legacy_sample_points': len(legacy_xyz),
                    'stratified_sample_points': len(strat_xyz), 'available_coverage_cells': available_cells,
                    'legacy_coverage_cells': old_cells, 'stratified_coverage_cells': new_cells,
                    'legacy_cell_coverage_fraction': float(old_cells / available_cells) if available_cells else None,
                    'stratified_cell_coverage_fraction': float(new_cells / available_cells) if available_cells else None,
                    'legacy_centroid_error_m_vs_masked_reference': legacy_mean_err,
                    'stratified_centroid_error_m_vs_masked_reference': strat_mean_err,
                    'legacy_covariance_error_m2_vs_masked_reference': legacy_cov_err,
                    'stratified_covariance_error_m2_vs_masked_reference': strat_cov_err,
                    'centroid_x_m': g['centroid_xyz_m'][0] if g['centroid_xyz_m'] else None,
                    'centroid_y_m': g['centroid_xyz_m'][1] if g['centroid_xyz_m'] else None,
                    'centroid_z_m': g['centroid_xyz_m'][2] if g['centroid_xyz_m'] else None,
                    'extent_x_m': g['extent_xyz_m'][0] if g['extent_xyz_m'] else None,
                    'extent_y_m': g['extent_xyz_m'][1] if g['extent_xyz_m'] else None,
                    'extent_z_m': g['extent_xyz_m'][2] if g['extent_xyz_m'] else None,
                    'pca_lambda1_m2': g['eigenvalues_m2'][0] if g['eigenvalues_m2'] else None,
                    'pca_lambda2_m2': g['eigenvalues_m2'][1] if g['eigenvalues_m2'] else None,
                    'pca_lambda3_m2': g['eigenvalues_m2'][2] if g['eigenvalues_m2'] else None,
                    'pca_axis_x': g['principal_axis_xyz'][0] if g['principal_axis_xyz'] else None,
                    'pca_axis_y': g['principal_axis_xyz'][1] if g['principal_axis_xyz'] else None,
                    'pca_axis_z': g['principal_axis_xyz'][2] if g['principal_axis_xyz'] else None,
                    'planarity_ratio': g['planarity_ratio'],
                    'median_reprojection_error_px': g['median_reprojection_error_px'],
                    'optical_centroid_step_m': None, 'optical_axis_step_deg_undirected': None,
                    'elapsed_sample_time_s': None}
                rows.append(row)
                if fi in extras:
                    out = a.output / 'previews' / f'{a.demo}_{fi:06d}_{cam}'
                    preview(rgb, strat_uv, strat_xyz, out.with_name(out.name + '_stratified.png'))
                    preview(rgb, legacy_uv, legacy_xyz, out.with_name(out.name + '_legacy.png'))
                    if a.save_preview_points:
                        np.savez_compressed(out.with_suffix('.npz'), xyz_optical_m=strat_xyz,
                            uv_pixels=strat_uv, intrinsics_K=K, mask=mask)
                print(f'{cam:16s} frame={fi:4d} mask={mask.sum():5d} '
                      f'points={len(strat_xyz):4d} cells={old_cells}/{new_cells}/{available_cells}')
    # Compare neighboring samples within *the same moving camera* for diagnostics only.
    # Do not equate changes in optical frame to motion of the cloth in world frame.
    for cam in cameras:
        sampled = sorted((r for r in rows if r['camera'] == cam and r['is_benchmark_sample']),
                         key=lambda r: r['frame_index'])
        prev = None
        for row in sampled:
            if prev is not None and row['centroid_x_m'] is not None and prev['centroid_x_m'] is not None:
                one = np.array([row[f'centroid_{k}_m'] for k in 'xyz'])
                two = np.array([prev[f'centroid_{k}_m'] for k in 'xyz'])
                row['optical_centroid_step_m'] = float(np.linalg.norm(one-two))
                axis1 = np.array([row[f'pca_axis_{k}'] for k in 'xyz'])
                axis2 = np.array([prev[f'pca_axis_{k}'] for k in 'xyz'])
                dot = np.clip(abs(float(np.dot(axis1, axis2))), 0., 1.)
                row['optical_axis_step_deg_undirected'] = float(np.degrees(np.arccos(dot)))
                row['elapsed_sample_time_s'] = float(row['time_s']-prev['time_s'])
            prev = row
    with (a.output/'sparse_3d_v2_metrics.csv').open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {'purpose': 'V2 optical-frame visibility diagnostics and 2048-point sampling A/B',
        'hdf5': str(a.hdf5), 'demo': a.demo, 'calibrations': [str(p) for p in a.calibration],
        'reference': 'All VALID DEPTH pixels within HSV mask, NOT garment ground truth',
        'sample_policy': 'Original v1 equally spaced row-major valid pixel indices vs grid-stratified',
        'max_points_per_camera_frame': a.max_points, 'strata_grid': a.strata_grid,
        'coverage_grid': a.coverage_grid, 'selected_frame_count': len(selected),
        'intrinsics': {c: camera_matrix(256, 256, DEFAULT_HFOV[c]).tolist() for c in cameras},
        'cameras': {}, 'warnings': [
            'Do not interpret image coverage or mask depth validity as true garment accuracy.',
            'Sampling errors are against all MASKED pixels, not real garment ground truth.',
            'Head/wrist optical-frame steps include camera motion; not cloth displacement.',
            'PCA 3D orientation has sign ambiguity and cannot recognize shirt front/back.',
            'No points from distinct cameras were merged; no camera extrinsics were used.',
            'Higher geometric coverage does not establish garment semantic correctness.']}
    for cam in cameras:
        rr = [r for r in rows if r['camera'] == cam and r['is_benchmark_sample']]
        vr = [r for r in rr if r['mask_valid'] and r['stratified_sample_points'] >= 3]
        lc, sc = median(vr, 'legacy_cell_coverage_fraction'), median(vr, 'stratified_cell_coverage_fraction')
        le, se = median(vr, 'legacy_centroid_error_m_vs_masked_reference'), median(vr, 'stratified_centroid_error_m_vs_masked_reference')
        covle, covse = median(vr, 'legacy_covariance_error_m2_vs_masked_reference'), median(vr, 'stratified_covariance_error_m2_vs_masked_reference')
        result['cameras'][cam] = {
            'benchmark_frames': len(rr), 'valid_masks': len(vr),
            'mask_valid_rate': len(vr)/len(rr) if rr else None,
            'median_mask_area_fraction': median(vr, 'mask_area_fraction'),
            'median_mask_border_fraction_4px': median(vr, 'mask_border_fraction_4px'),
            'fraction_valid_masks_touching_image_edge': float(np.mean([r['mask_touches_image_edge'] for r in vr])) if vr else None,
            'median_mask_components': median(vr, 'mask_components'),
            'median_mask_depth_valid_fraction': median(vr, 'mask_depth_valid_fraction'),
            'median_legacy_cell_coverage': lc, 'median_stratified_cell_coverage': sc,
            'median_legacy_centroid_error_m': le, 'median_stratified_centroid_error_m': se,
            'median_legacy_covariance_error_m2': covle,
            'median_stratified_covariance_error_m2': covse,
            'median_optical_centroid_step_m_camera_motion_not_removed': median(vr, 'optical_centroid_step_m'),
            'median_optical_pca_axis_step_deg_camera_motion_not_removed': median(vr, 'optical_axis_step_deg_undirected'),
            'stratified_improves_median_coverage': None if lc is None or sc is None else bool(sc > lc),
            'stratified_reduces_median_centroid_error': None if le is None or se is None else bool(se < le),
            'stratified_reduces_median_covariance_error': None if covle is None or covse is None else bool(covse < covle),
        }
    result['linux_maxrss_reported_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (a.output/'sparse_3d_v2_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved', a.output/'sparse_3d_v2_summary.json')
    print('Saved', a.output/'sparse_3d_v2_metrics.csv')
    print('Peak RSS MiB', round(result['linux_maxrss_reported_kib']/1024, 2))


if __name__ == '__main__':
    main()
