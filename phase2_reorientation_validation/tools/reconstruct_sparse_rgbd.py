#!/usr/bin/env python3
"""PhyRC sparse masked RGB-D reconstruction in each camera optical frame.

Only policy.hdf5 /obs and public camera FOV configuration are used.
No audit camera poses, privileged simulator state, or camera-to-world transforms.
Requires _common.py in the same validation tools folder.
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
from phase2_reorientation.perception import CameraFrame, MultiHSVRangeSegmenter, PerceptionProcessor

DEFAULT_HFOV = {'overview': 60., 'robot_0_wrist': 70., 'robot_1_wrist': 70.,
                'robot_0_head': 70., 'robot_1_head': 70.}


def camera_matrix(w: int, h: int, hfov_deg: float) -> np.ndarray:
    if not 0 < hfov_deg < 180:
        raise ValueError('Horizontal FOV must be in (0,180) degrees')
    f = w / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    return np.array([[f, 0., w / 2.], [0., f, h / 2.], [0., 0., 1.]], dtype=np.float64)


def sparse_points(mask: np.ndarray, depth: np.ndarray, valid: np.ndarray, K: np.ndarray,
                  max_points: int, min_depth: float, max_depth: float):
    """Deterministic spatially spread pixel subsampling; result in optical XYZ metres."""
    good = (mask & valid & np.isfinite(depth) & (depth > min_depth) &
            (depth < max_depth))
    flat = np.flatnonzero(good)
    n_available = len(flat)
    if not n_available:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32), 0
    # Evenly cover valid mask pixels by row-major index; reproducible, no RNG.
    if n_available > max_points:
        ix = (np.arange(max_points, dtype=np.int64) * n_available) // max_points
        flat = flat[ix]
    vv, uu = np.unravel_index(flat, mask.shape)
    z = depth[vv, uu].astype(np.float64)
    x = (uu.astype(np.float64) - K[0, 2]) * z / K[0, 0]
    y = (vv.astype(np.float64) - K[1, 2]) * z / K[1, 1]
    xyz = np.column_stack((x, y, z)).astype(np.float32)
    uv = np.column_stack((uu, vv)).astype(np.int32)
    return xyz, uv, n_available


def geometry(xyz: np.ndarray, uv: np.ndarray, K: np.ndarray) -> dict:
    n = len(xyz)
    if n < 3:
        return {'centroid_xyz_m': None, 'extent_xyz_m': None,
                'eigenvalues_m2': None, 'principal_axis_xyz': None,
                'planarity_ratio': None, 'median_reprojection_error_px': None,
                'median_z_m': None}
    pts = xyz.astype(np.float64)
    centroid = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.)
    axis = vecs[:, order[0]]
    # Axis is *undirected*: deterministic sign for stable logs, NOT garment front/back.
    if axis[np.argmax(np.abs(axis))] < 0:
        axis *= -1
    z = pts[:, 2]
    u_reproj = pts[:, 0] * K[0, 0] / z + K[0, 2]
    v_reproj = pts[:, 1] * K[1, 1] / z + K[1, 2]
    reproj = np.hypot(u_reproj - uv[:, 0], v_reproj - uv[:, 1])
    return {
        'centroid_xyz_m': centroid.tolist(),
        'extent_xyz_m': (np.quantile(pts, .9, axis=0) - np.quantile(pts, .1, axis=0)).tolist(),
        'eigenvalues_m2': vals.tolist(),
        'principal_axis_xyz': axis.tolist(),
        'planarity_ratio': float(vals[-1] / vals[0]) if vals[0] > 1e-12 else None,
        'median_reprojection_error_px': float(np.median(reproj)),
        'median_z_m': float(np.median(z)),
    }


def preview(rgb: np.ndarray, uv: np.ndarray, xyz: np.ndarray, path: Path):
    img = Image.fromarray(rgb, 'RGB').convert('RGBA')
    mask_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(mask_layer)
    if len(uv):
        z = xyz[:, 2]
        lo, hi = np.quantile(z, [.05, .95])
        denom = max(float(hi - lo), 1e-9)
        scaled = np.clip((z - lo) / denom, 0, 1)
        for (u, v), s in zip(uv, scaled):
            # Warm=near, cool=far (within per-image sampling depth range).
            drawer.point((int(u), int(v)), fill=(int(255 * (1-s)), 70, int(255*s), 230))
    img = Image.alpha_composite(img, mask_layer).convert('RGB')
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5', type=Path, required=True)
    parser.add_argument('--demo', default='demo_0')
    parser.add_argument('--calibration', type=Path, action='append', required=True)
    parser.add_argument('--camera', choices=('all',) + CAMERAS, default='all')
    parser.add_argument('--every', type=int, default=50)
    parser.add_argument('--max-frames', type=int, default=30)
    parser.add_argument('--max-points', type=int, default=2048)
    parser.add_argument('--min-component-pixels', type=int, default=80)
    parser.add_argument('--keep-components', type=int, default=3)
    parser.add_argument('--min-mask-pixels', type=int, default=30)
    parser.add_argument('--min-depth', type=float, default=0.)
    parser.add_argument('--preview-frames', type=int, nargs='*', default=[0, 600, 1250, 1850, 2600])
    parser.add_argument('--save-preview-points', action='store_true',
                        help='Save xyz/uv sample NPZ for preview frames only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for attr in ('every', 'max_frames', 'max_points', 'min-component-pixels', 'keep-components', 'min-mask-pixels'):
        value = getattr(args, attr.replace('-', '_'))
        if value < 1:
            parser.error(f'{attr} must be >= 1')
    if len(args.calibration) < 2:
        parser.error('Pass both blue v2 and yellow v3 --calibration JSON files')
    if not args.hdf5.is_file() or any(not p.is_file() for p in args.calibration):
        parser.error('Missing HDF5 or calibration JSON file')
    ranges = []
    for path in args.calibration:
        c = json.loads(path.read_text(encoding='utf-8'))
        ranges.append((tuple(c['hsv_lower']), tuple(c['hsv_upper'])))
    segmenter = MultiHSVRangeSegmenter(hsv_ranges=tuple(ranges),
                                      min_component_pixels=args.min_component_pixels,
                                      keep_largest_components=args.keep_components)
    processor = PerceptionProcessor(segmenter=segmenter, min_mask_pixels=args.min_mask_pixels)
    cameras = CAMERAS if args.camera == 'all' else (args.camera,)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with h5py.File(args.hdf5, 'r') as h5:
        obs = h5[f'data/{args.demo}/obs']
        sources = {cam: get_camera_arrays(obs, cam) for cam in cameras}
        n = min(ds[0].shape[0] for ds in sources.values())
        candidates = list(range(0, n, args.every))
        if len(candidates) > args.max_frames:
            indexes = np.linspace(0, len(candidates) - 1, args.max_frames, dtype=int)
            selected = {candidates[i] for i in indexes}
        else:
            selected = set(candidates)
        extras = {f for f in args.preview_frames if 0 <= f < n}
        for fi in sorted(selected | extras):
            for cam in cameras:
                rgb_ds, depth_ds, valid_ds = sources[cam]
                rgb = rgb_ds[fi].astype(np.uint8)
                depth = read_depth_frame(depth_ds, fi).astype(np.float32)
                valid = read_depth_frame(valid_ds, fi).astype(bool)
                h, w = rgb.shape[:2]
                K = camera_matrix(w, h, DEFAULT_HFOV[cam])
                detection = processor.process_frame(CameraFrame(
                    camera_name=cam, rgb=rgb, depth_m=depth, depth_valid=valid))
                # From contract: optical-axis Z, camera-specific clip range in metres.
                max_depth = 10. if cam == 'overview' else (3. if 'wrist' in cam else 5.)
                xyz, uv, n_available = sparse_points(
                    detection.mask, depth, valid, K, args.max_points,
                    args.min_depth, max_depth)
                g = geometry(xyz, uv, K)
                row = {'camera': cam, 'frame_index': fi, 'time_s': read_time(obs, fi),
                       'is_benchmark_sample': int(fi in selected),
                       'mask_valid': int(detection.valid),
                       'mask_area_px': int(detection.area_px),
                       'depth_valid_mask_points': n_available,
                       'sparse_points': len(xyz),
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
                       'median_reprojection_error_px': g['median_reprojection_error_px']}
                rows.append(row)
                if fi in extras:
                    base = args.output / 'previews' / f'{args.demo}_{fi:06d}_{cam}'
                    preview(rgb, uv, xyz, base.with_suffix('.png'))
                    if args.save_preview_points:
                        np.savez_compressed(base.with_suffix('.npz'), xyz_optical_m=xyz,
                                            uv_pixels=uv, intrinsics_K=K)
                print(f'{cam:16s} frame={fi:4d} valid={int(detection.valid)} '
                      f'mask={int(detection.area_px):5d} xyz={len(xyz):4d} '
                      f'centroid_z={row["centroid_z_m"]}')
    if not rows:
        raise RuntimeError('No frames processed')
    with (args.output / 'sparse_3d_metrics.csv').open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'purpose': 'Sparse 3D geometry in individual optical camera frames only',
              'source': str(args.hdf5), 'calibrations': [str(p) for p in args.calibration],
              'camera_frame': 'optical: +X right, +Y down, +Z forward; metres',
              'intrinsics': {cam: camera_matrix(256, 256, DEFAULT_HFOV[cam]).tolist() for cam in cameras},
              'depth': 'optical-axis Z in metres; nonpositive/invalid/nonfinite excluded',
              'downsample': 'deterministic row-major even-index subset of valid masked pixels',
              'max_points_per_camera_frame': args.max_points,
              'frames_used_for_summary': 'benchmark samples only; extra preview frames excluded',
              'cameras': {}, 'warnings': ['HSV mask is not segmentation ground truth.',
                                        'PCA axis sign is not garment front/back.',
                                        'Reprojection error only checks internal arithmetic, not calibration accuracy.',
                                        'Measurements from different cameras must not be merged without extrinsics.']}
    for cam in cameras:
        subset = [r for r in rows if r['camera'] == cam and r['is_benchmark_sample']]
        valid = [r for r in subset if r['mask_valid'] and r['sparse_points'] >= 3]
        def med(key):
            vals = [r[key] for r in valid if r[key] is not None]
            return float(np.median(vals)) if vals else None
        result['cameras'][cam] = {'sample_frames': len(subset),
                                  'mask_valid_rate': float(np.mean([r['mask_valid'] for r in subset])),
                                  'frames_with_xyz': sum(r['sparse_points'] >= 3 for r in subset),
                                  'median_sparse_points_valid_mask': med('sparse_points'),
                                  'median_centroid_z_m': med('centroid_z_m'),
                                  'median_extent_z_m': med('extent_z_m'),
                                  'median_planarity_ratio': med('planarity_ratio'),
                                  'median_reprojection_error_px': med('median_reprojection_error_px')}
    result['linux_maxrss_reported_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = args.output / 'sparse_3d_summary.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved:', args.output / 'sparse_3d_metrics.csv')
    print('Saved:', out)
    print('Linux max RSS:', round(result['linux_maxrss_reported_kib'] / 1024, 2), 'MiB')


if __name__ == '__main__':
    main()
