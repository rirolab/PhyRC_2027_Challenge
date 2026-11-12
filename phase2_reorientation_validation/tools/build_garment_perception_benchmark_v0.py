#!/usr/bin/env python3
"""Export a reproducible actor-observation-only PhyRC garment perception benchmark.

This is STEP 1 of a staged model integration: it makes RGB, existing HSV
predictions, optional masked depth, a per-image manifest and an empty human
annotation sheet. HSV predictions are NEVER treated as ground truth.

Install beside _common.py inside phase2_reorientation_validation/tools.
"""
from __future__ import annotations
import argparse
import csv
import json
import resource
from pathlib import Path
import h5py
import numpy as np
from PIL import Image

from _common import CAMERAS, get_camera_arrays, read_depth_frame, read_time
from phase2_reorientation.perception import CameraFrame, PerceptionProcessor
try:
    from phase2_reorientation.perception import MultiHSVRangeSegmenter
except ImportError:
    # Compatibility for earlier local source bundles lacking MultiHSVRangeSegmenter.
    # VM versions that contain the established class use it directly.
    from phase2_reorientation.perception import HSVRangeSegmenter
    from collections import deque
    class MultiHSVRangeSegmenter:
        def __init__(self, hsv_ranges, min_component_pixels=80, keep_largest_components=3):
            self.children = [HSVRangeSegmenter(hsv_lower=lo, hsv_upper=hi) for lo,hi in hsv_ranges]
            self.min_pixels = min_component_pixels
            self.keep = keep_largest_components
        def segment(self, rgb, depth_m, depth_valid, camera_name):
            raw = np.zeros(rgb.shape[:2], dtype=bool)
            for model in self.children:
                raw |= model.segment(rgb, depth_m, depth_valid, camera_name)
            seen = np.zeros_like(raw)
            h,w=raw.shape
            clusters=[]
            for y,x in zip(*np.nonzero(raw)):
                if seen[y,x]: continue
                seen[y,x]=True
                q=deque([(int(y),int(x))]); component=[]
                while q:
                    cy,cx=q.popleft();component.append((cy,cx))
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            ny,nx=cy+dy,cx+dx
                            if 0<=ny<h and 0<=nx<w and raw[ny,nx] and not seen[ny,nx]:
                                seen[ny,nx]=True;q.append((ny,nx))
                if len(component)>=self.min_pixels: clusters.append(component)
            clusters.sort(key=len, reverse=True)
            filtered=np.zeros_like(raw)
            for cluster in clusters[:self.keep]:
                for y,x in cluster: filtered[y,x]=True
            return filtered

DEFAULT_FRAMES = [0, 600, 1000, 1250, 1500, 1600, 1700, 1750, 1850, 1950, 2050, 2150, 2200, 2300, 2400, 2600]
FIELDS = ['sample_id', 'demo', 'frame', 'time_s', 'camera', 'rgb', 'hsv_prediction',
          'depth_valid', 'mask_area_px', 'mask_depth_valid_fraction',
          'review_priority', 'human_gt_mask', 'collar_xy', 'left_sleeve_xy',
          'right_sleeve_xy', 'hem_left_xy', 'hem_right_xy',
          'front_back_label', 'orientation_observable', 'review_notes']


def get_config(p):
    d = json.loads(p.read_text(encoding='utf-8'))
    if 'hsv_lower' not in d or 'hsv_upper' not in d:
        raise ValueError(f'Calibration is missing hsv_lower/hsv_upper: {p}')
    return tuple(d['hsv_lower']), tuple(d['hsv_upper'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hdf5', type=Path, required=True)
    ap.add_argument('--demo', default='demo_0')
    ap.add_argument('--calibration', type=Path, action='append', required=True)
    ap.add_argument('--frames', type=int, nargs='+', default=DEFAULT_FRAMES)
    ap.add_argument('--camera', action='append', choices=CAMERAS, help='Repeat for subset; omitted=all five')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--save-depth', action='store_true', help='Save metric optical Z and validity in compressed npz per sample')
    ap.add_argument('--min-component-pixels', type=int, default=80)
    ap.add_argument('--keep-components', type=int, default=3)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f'Output directory already contains files: {args.output} (choose a new output directory)')
    cams = tuple(dict.fromkeys(args.camera or CAMERAS))
    bounds = tuple(get_config(p) for p in args.calibration)
    processor = PerceptionProcessor(
        segmenter=MultiHSVRangeSegmenter(hsv_ranges=bounds,
              min_component_pixels=args.min_component_pixels,
              keep_largest_components=args.keep_components), min_mask_pixels=30)
    args.output.mkdir(parents=True, exist_ok=True)
    for sub in ('rgb', 'hsv_prediction', 'gt_masks_manual'):
        (args.output/sub).mkdir(exist_ok=True)
    if args.save_depth:
        (args.output/'depth').mkdir(exist_ok=True)
    rows = []
    with h5py.File(args.hdf5, 'r') as f:
        group = f'data/{args.demo}/obs'
        if group not in f:
            raise ValueError(f'HDF5 missing {group}')
        obs = f[group]
        sources = {cam: get_camera_arrays(obs, cam) for cam in cams}
        size = len(next(iter(sources.values()))[0])
        indices = sorted(set(args.frames))
        if not indices or any(i < 0 or i >= size for i in indices):
            raise ValueError(f'Invalid frame index; valid range is 0..{size-1}')
        for idx in indices:
            for cam in cams:
                rgb_ds, depth_ds, valid_ds = sources[cam]
                rgb = np.asarray(rgb_ds[idx], dtype=np.uint8)
                depth = np.asarray(read_depth_frame(depth_ds, idx), dtype=np.float32)
                valid = np.asarray(read_depth_frame(valid_ds, idx), dtype=bool)
                det = processor.process_frame(CameraFrame(camera_name=cam, rgb=rgb, depth_m=depth, depth_valid=valid))
                mask = np.asarray(det.mask, dtype=bool)
                name = f'{args.demo}_{idx:06d}_{cam}'
                rgb_path = Path('rgb') / (name + '.png')
                mask_path = Path('hsv_prediction') / (name + '.png')
                Image.fromarray(rgb, mode='RGB').save(args.output/rgb_path)
                Image.fromarray(mask.astype(np.uint8)*255, mode='L').save(args.output/mask_path)
                good = valid & np.isfinite(depth) & (depth > 0)
                if args.save_depth:
                    np.savez_compressed(args.output/'depth'/(name+'.npz'), depth_m=depth, depth_valid=valid)
                frac = float(np.count_nonzero(mask & good)/max(1, int(mask.sum())))
                rows.append({
                    'sample_id': name, 'demo': args.demo, 'frame':idx,
                    'time_s':round(read_time(obs, idx), 7), 'camera':cam,
                    'rgb':str(rgb_path), 'hsv_prediction':str(mask_path),
                    'depth_valid':str(Path('depth')/(name+'.npz')) if args.save_depth else '',
                    'mask_area_px':int(mask.sum()), 'mask_depth_valid_fraction':round(frac, 6),
                    'review_priority':'high' if idx in (1250, 1500, 1600, 2600) else 'normal',
                    'human_gt_mask':'', 'collar_xy':'', 'left_sleeve_xy':'',
                    'right_sleeve_xy':'', 'hem_left_xy':'', 'hem_right_xy':'',
                    'front_back_label':'unknown', 'orientation_observable':'unknown',
                    'review_notes':''})
            print(f'exported frame {idx}: {len(cams)} camera images')
    with (args.output/'annotation_manifest.csv').open('w', encoding='utf-8', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
    summary = {
        'purpose':'STEP 1: reproducible RGB-D garment segmentation/keypoint validation set',
        'source':str(args.hdf5), 'demo':args.demo, 'frames':indices, 'cameras':cams,
        'samples':len(rows), 'hsv_calibrations':[str(p) for p in args.calibration],
        'policy_observations_only':True,
        'rules':['HSV masks are baseline model predictions, NOT ground truth.',
                 'Human labels must be independently drawn from public RGB observations.',
                 'No audit, cloth mesh, ground truth segmentation or privileged poses.',
                 'All samples originate from one episode; DO NOT claim independent test generalization.',
                 'Use separate episodes for final held-out testing.',
                 'Unknown/occluded landmarks must remain unlabeled rather than guessed.'],
        'peak_RSS_KiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    }
    (args.output/'export_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False), encoding='utf-8')
    print('Saved:',args.output, '|',len(rows),'samples')


if __name__ == '__main__':
    main()
