#!/usr/bin/env python3
"""PhyRC Garment Observation State v0 — offline, optical-frame, actor-observation only.

Depends on _common.py and reconstruct_sparse_rgbd.py (v1) in the same folder.
Retains v1 uniform row-major sampler and HSV blue-v2/yellow-v3 segmentation.
Does NOT fuse camera XYZ, assume world alignment, label shirt front/back, or use audit.
The camera ranking is an UNCALIBRATED observation-quality heuristic, not garment confidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import resource
from pathlib import Path

import h5py
import numpy as np

from _common import CAMERAS, get_camera_arrays, read_depth_frame, read_time
from reconstruct_sparse_rgbd import DEFAULT_HFOV, camera_matrix, sparse_points, geometry
from phase2_reorientation.perception import CameraFrame, MultiHSVRangeSegmenter, PerceptionProcessor

CAMERA_CLIP_MAX = {'overview': 10., 'robot_0_wrist': 3., 'robot_1_wrist': 3.,
                   'robot_0_head': 5., 'robot_1_head': 5.}


def mask_features(mask: np.ndarray, valid_depth_mask: np.ndarray, grid: int = 16) -> dict:
    """Cheap mask diagnostics. Use baseline segmenter post-CC mask."""
    h, w = mask.shape
    y, x = np.nonzero(mask)
    area = len(x)
    good = int(np.count_nonzero(mask & valid_depth_mask))
    if not area:
        return dict(area_px=0, area_fraction=0., depth_fraction=None,
                    edge_touch=False, border_fraction=0., bbox_xyxy=None,
                    occupied_cells=0, largest_component_fraction=None, component_count=0)
    edge = bool((x == 0).any() or (x == w-1).any() or (y == 0).any() or (y == h-1).any())
    bw = 4
    border_count = int(np.count_nonzero((x < bw) | (x >= w-bw) | (y < bw) | (y >= h-bw)))
    # Use cv2 if installed? No: retain same no-extra-dependency connected component logic.
    # 8-connectivity implementation in v2; only imports this exact helper when available.
    from reconstruct_sparse_rgbd_v2 import components8
    sizes = components8(mask)
    cells = np.unique((y * grid // h) * grid + (x * grid // w)).size
    return dict(area_px=area, area_fraction=float(area/(h*w)),
                depth_fraction=float(good/area), edge_touch=edge,
                border_fraction=float(border_count/area),
                bbox_xyxy=[int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                occupied_cells=int(cells), component_count=len(sizes),
                largest_component_fraction=float(sizes[0]/area) if sizes else None)


def observation_quality(detection_valid: bool, count: int, f: dict, min_points: int = 64):
    """Uncalibrated, per-camera *observation* score, not garment-state probability.

    Deliberately no camera identity bias; no preference for larger masks (background
    spill could increase mask area). Thresholds are provisional and must be tuned.
    """
    if not detection_valid or count < 3 or not f['area_px']:
        return 'unavailable', 0.0, ['invalid_mask_or_insufficient_xyz']
    reasons = []
    if count < min_points:
        reasons.append('few_xyz_points')
    if f['edge_touch']:
        reasons.append('mask_touches_fov_edge')
    if f['component_count'] > 1:
        reasons.append('fragmented_mask')
    if f['depth_fraction'] is not None and f['depth_fraction'] < .9:
        reasons.append('incomplete_depth')
    if f['area_fraction'] < .005:
        reasons.append('small_mask')
    # Score strictly heuristic: cannot substitute for segmentation GT.
    point_term = min(1., count / max(min_points, 1))
    depth_term = max(0., min(1., f['depth_fraction'] or 0.))
    edge_term = 0.65 if f['edge_touch'] else 1.
    fragment_term = 1. / (1. + .12 * max(0, f['component_count'] - 1))
    size_term = min(1., f['area_fraction']/.005)
    score = float(point_term * depth_term * edge_term * fragment_term * size_term)
    # 'partial' explicitly preserves uncertainty for FOV cut/fragmentation.
    label = 'limited' if count < min_points or f['edge_touch'] or f['depth_fraction'] < .9 else 'observed_partial_unknown'
    return label, score, reasons


def safe_float(value):
    return None if value is None or not np.isfinite(float(value)) else float(value)


def median_valid(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hdf5', type=Path, required=True)
    p.add_argument('--demo', default='demo_0')
    p.add_argument('--calibration', action='append', type=Path, required=True)
    p.add_argument('--every', type=int, default=50)
    p.add_argument('--max-frames', type=int, default=30)
    p.add_argument('--max-points', type=int, default=2048)
    p.add_argument('--min-component-pixels', type=int, default=80)
    p.add_argument('--keep-components', type=int, default=3)
    p.add_argument('--min-mask-pixels', type=int, default=30)
    p.add_argument('--min-xyz-points', type=int, default=64)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for k in ['every','max_frames','max_points','min_component_pixels','keep_components',
              'min_mask_pixels','min_xyz_points']:
        if getattr(a,k) <= 0: p.error('--'+k.replace('_','-')+' must be positive')
    if len(a.calibration) != 2:
        p.error('Provide exactly 2 --calibration JSON files (blue v2 and yellow v3)')
    if not a.hdf5.is_file() or any(not f.is_file() for f in a.calibration):
        p.error('HDF5 or HSV calibration not found')
    from reconstruct_sparse_rgbd_v2 import components8  # dependency availability check
    del components8
    ranges = []
    for path in a.calibration:
        c = json.loads(path.read_text(encoding='utf8'))
        ranges.append((tuple(c['hsv_lower']),tuple(c['hsv_upper'])))
    segmenter = MultiHSVRangeSegmenter(hsv_ranges=tuple(ranges),
        min_component_pixels=a.min_component_pixels, keep_largest_components=a.keep_components)
    processor = PerceptionProcessor(segmenter=segmenter,min_mask_pixels=a.min_mask_pixels)
    a.output.mkdir(parents=True, exist_ok=True)
    all_rows = []; frame_states = []
    with h5py.File(a.hdf5, 'r') as h5:
        obs = h5[f'data/{a.demo}/obs']
        sources = {cam:get_camera_arrays(obs,cam) for cam in CAMERAS}
        n = min(v[0].shape[0] for v in sources.values())
        candidate = list(range(0,n,a.every))
        if len(candidate) > a.max_frames:
            selected = sorted(set(candidate[j] for j in np.linspace(0,len(candidate)-1,a.max_frames,dtype=int)))
        else: selected = candidate
        for fi in selected:
            items = []
            for cam in CAMERAS:
                r, d, v = sources[cam]
                rgb = r[fi].astype(np.uint8)
                depth = read_depth_frame(d,fi).astype(np.float32)
                valid = read_depth_frame(v,fi).astype(bool)
                h,w = rgb.shape[:2]
                K = camera_matrix(w,h,DEFAULT_HFOV[cam])
                detect = processor.process_frame(CameraFrame(camera_name=cam, rgb=rgb,
                                                depth_m=depth,depth_valid=valid))
                mask = np.asarray(detect.mask,dtype=bool)
                good = valid & np.isfinite(depth) & (depth>0) & (depth<CAMERA_CLIP_MAX[cam])
                xyz, uv, available = sparse_points(mask,depth,valid,K,a.max_points,0.,CAMERA_CLIP_MAX[cam])
                geo = geometry(xyz,uv,K)
                f = mask_features(mask,good)
                label, score, why = observation_quality(bool(detect.valid),len(xyz),f,a.min_xyz_points)
                item = {'camera':cam,'status':label,'quality_heuristic':score,
                        'quality_reasons':why,'xyz_points':int(len(xyz)),
                        'available_mask_depth_points':int(available),
                        'mask_area_fraction':f['area_fraction'],
                        'mask_area_px':f['area_px'],
                        'mask_edge_touch':f['edge_touch'],
                        'mask_border_fraction_4px':f['border_fraction'],
                        'mask_components':f['component_count'],
                        'largest_component_fraction':f['largest_component_fraction'],
                        'mask_depth_valid_fraction':f['depth_fraction'],
                        'mask_occupied_grid_cells':f['occupied_cells'],
                        'bbox_xyxy':f['bbox_xyxy'],
                        'optical_centroid_xyz_m':geo['centroid_xyz_m'],
                        'optical_extent_p10_p90_xyz_m':geo['extent_xyz_m'],
                        'optical_pca_eigenvalues_m2':geo['eigenvalues_m2'],
                        'optical_principal_axis_undirected':geo['principal_axis_xyz'],
                        'planarity_ratio':geo['planarity_ratio']}
                items.append(item)
                flat = {'frame':fi,'time_s':read_time(obs,fi),'camera':cam,**{
                    k:v for k,v in item.items() if not isinstance(v,(list,dict))}}
                for stem, seq in [('centroid',geo['centroid_xyz_m']),
                                  ('extent',geo['extent_xyz_m']),
                                  ('eigenvalue',geo['eigenvalues_m2']),
                                  ('pca_axis',geo['principal_axis_xyz'])]:
                    for ind,axis in enumerate('xyz'):
                        flat[f'{stem}_{axis}']=seq[ind] if seq else None
                all_rows.append(flat)
            # No optical-camera XYZ comparisons or fusion: only rank observational candidates.
            eligible = [x for x in items if x['status'] != 'unavailable']
            ranked = sorted(eligible,key=lambda x:(-x['quality_heuristic'],CAMERAS.index(x['camera'])))
            primary = ranked[0]['camera'] if ranked else None
            frame_states.append({'frame':fi,'time_s':read_time(obs,fi),
                'selected_observation_camera':primary,
                'selection_basis':'uncalibrated visibility heuristic, not physical garment confidence',
                'candidate_order':[x['camera'] for x in ranked],
                'has_3d_garment_pose':False,
                'cameras':items})
            print(f'frame={fi:4d} camera_candidate={primary or "NONE"} '
                  f'available={len(eligible)}/{len(CAMERAS)}')
    with (a.output/'garment_observation_frames.jsonl').open('w',encoding='utf8') as fp:
        for state in frame_states: fp.write(json.dumps(state,allow_nan=False)+'\n')
    with (a.output/'garment_observation_metrics.csv').open('w',newline='',encoding='utf8') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(all_rows[0]))
        writer.writeheader();writer.writerows(all_rows)
    by_camera={}
    for cam in CAMERAS:
        rr=[r for r in all_rows if r['camera']==cam]
        by_camera[cam]={'samples':len(rr),
            'available_3d_fraction':float(np.mean([r['status']!='unavailable' for r in rr])),
            'edge_touch_fraction_of_available':float(np.mean([r['mask_edge_touch'] for r in rr if r['status']!='unavailable'])) if any(r['status']!='unavailable' for r in rr) else None,
            'median_quality_heuristic_available':median_valid([r for r in rr if r['status']!='unavailable'],'quality_heuristic'),
            'median_mask_fraction_available':median_valid([r for r in rr if r['status']!='unavailable'],'mask_area_fraction'),
            'selected_count':sum(x['selected_observation_camera']==cam for x in frame_states)}
    counts = {cam:by_camera[cam]['selected_count'] for cam in CAMERAS}
    summary={'purpose':'Garment Observation State v0 - camera-specific RGB-D features and observability selection',
        'input_data':str(a.hdf5),'demo':a.demo,'calibrations':[str(p) for p in a.calibration],
        'sample_count':len(frame_states),'max_points_per_camera_frame':a.max_points,
        'sampler':'v1 evenly-spaced row-major pixel index; stratified sampling deliberately excluded',
        'optical_frame':'X right, Y down, Z forward; metres',
        'ranker':'uncalibrated heuristic: low point count/depth incompleteness/edge touch/mask fragmentation/small mask penalized; no GT calibration',
        'selection_counts':counts,'frames_no_valid_observation':sum(x['selected_observation_camera'] is None for x in frame_states),
        'per_camera':by_camera,
        'known_limits':['camera selection is NOT a calibrated success metric',
            'HSV segmentation has no verified semantic ground truth',
            'optical XYZ across different cameras are NOT comparable or merged',
            'moving camera XYZ changes cannot be identified as cloth motion',
            'PCA sign does not encode garment front/back; no garment world pose estimated',
            'mask area and depth-validity alone cannot establish true garment coverage'],
        'linux_maxrss_reported_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (a.output/'garment_observation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')
    print('Saved',a.output/'garment_observation_summary.json')
    print('Peak RSS MiB',round(summary['linux_maxrss_reported_kib']/1024,2))


if __name__=='__main__':
    main()
