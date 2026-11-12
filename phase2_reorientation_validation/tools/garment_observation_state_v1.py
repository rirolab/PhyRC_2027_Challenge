#!/usr/bin/env python3
"""PhyRC observation-state v1: global/local camera ranking and v0 A/B.

Consumes exact per-frame actor-only features emitted by garment_observation_state_v0.py.
No audit, world-camera poses, simulator privileged state or cross-camera XYZ fusion.
Scores are **uncalibrated observation-utility heuristics**, not garment accuracy.
Optional --hdf5/--calibration creates RGB/mask diagnostic sheets for selected frames.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import resource
from collections import Counter
from pathlib import Path

CAMERAS = ('overview', 'robot_0_wrist', 'robot_1_wrist', 'robot_0_head', 'robot_1_head')
GLOBAL = ('overview', 'robot_0_head', 'robot_1_head')
LOCAL = ('robot_0_wrist', 'robot_1_wrist')


def clamp01(x): return max(0., min(1., float(x)))


def rank_feature(x, role):
    """Returns (available, score, evidence), without learned/GT calibration.

    Scores are *within-role*, never compare across global and local roles.
    Camera identity is used ONLY for role eligibility, not quality bonuses.
    Fullness is unknown: mask area is merely supportive pixel evidence.
    """
    evidence = []
    if x.get('status') == 'unavailable' or int(x.get('xyz_points') or 0) < 64:
        return False, 0., ['insufficient_masked_depth_points']
    area = float(x.get('mask_area_fraction') or 0.)
    depth = clamp01(x.get('mask_depth_valid_fraction') or 0.)
    components = int(x.get('mask_components') or 0)
    count = int(x.get('xyz_points') or 0)
    edge = bool(x.get('mask_edge_touch'))
    border = clamp01(x.get('mask_border_fraction_4px') or 0.)
    cells = int(x.get('mask_occupied_grid_cells') or 0)
    if role == 'global':
        # A tiny clean patch MUST NOT saturate quality like v0's 0.005 threshold.
        if area < .008:
            return False, 0., ['global_mask_below_0.8pct_of_image']
        area_support = clamp01(area / .08)
        spatial_support = clamp01(cells / 40.)
        score = (.42 * area_support + .23 * spatial_support +
                 .17 * (0.70 if edge else 1.0) +
                 .10 * (1. / (1. + .12 * max(0, components - 1))) +
                 .08 * depth)
        if area < .02: evidence.append('weak_global_mask_area')
        if edge: evidence.append('garment_mask_at_fov_boundary')
    else:
        # Wrist crops are expected; edge touch is diagnostic, not disqualifying.
        if area < .003:
            return False, 0., ['local_mask_below_0.3pct_of_image']
        area_support = clamp01(area / .06)
        # Border fraction has a mild penalty; avoid penalizing crop merely for touch.
        score = (.38 * area_support + .24 * clamp01(count / 1024.) +
                 .18 * depth + .12 * (1. - .4 * border) +
                 .08 * (1. / (1. + .1 * max(0, components - 1))))
        if edge: evidence.append('local_view_truncated_expected')
        if border > .2: evidence.append('substantial_crop_at_border')
    if components > 1: evidence.append('fragmented_hsv_mask')
    if depth < .95: evidence.append('incomplete_mask_depth')
    return True, round(clamp01(score), 6), evidence


def choose(items, role):
    rows = []
    eligible_ids = GLOBAL if role == 'global' else LOCAL
    for item in items:
        if item['camera'] not in eligible_ids: continue
        ok, score, reason = rank_feature(item, role)
        rows.append({'camera':item['camera'], 'available':ok, 'score':score,
                     'reasons':reason, 'mask_area_fraction':item['mask_area_fraction'],
                     'mask_edge_touch':item['mask_edge_touch'],
                     'xyz_points':item['xyz_points']})
    rows.sort(key=lambda r: (-int(r['available']), -r['score'], eligible_ids.index(r['camera'])))
    primary = next((r['camera'] for r in rows if r['available']), None)
    # Even available is NOT a guarantee of meaningful garment shape/semantics.
    return primary, rows


def prepare(v0_states):
    records, metrics = [], []
    for state in v0_states:
        fi = int(state['frame'])
        if len(state['cameras']) != 5 or {c['camera'] for c in state['cameras']} != set(CAMERAS):
            raise ValueError(f'Frame {fi} does not contain exactly five expected cameras')
        g, gr = choose(state['cameras'], 'global')
        l, lr = choose(state['cameras'], 'local')
        record = {'frame':fi, 'time_s':state.get('time_s'), 'v0_selected':state.get('selected_observation_camera'),
                  'global_selected':g,'local_selected':l,
                  'global_ranked':gr,'local_ranked':lr,
                  'global_observation_available':g is not None,
                  'local_observation_available':l is not None,
                  'has_world_pose':False,'has_front_back_label':False,
                  'selection_basis':'Uncalibrated, separate global/local observation utility; no garment GT'}
        records.append(record)
        for role, ranking in [('global',gr), ('local',lr)]:
            for r in ranking:
                metrics.append({'frame':fi,'role':role,'camera':r['camera'],
                                'available':r['available'],'v1_score':r['score'],
                                'selected':r['camera']==(g if role=='global' else l),
                                'v0_selected':r['camera']==state.get('selected_observation_camera'),
                                'mask_fraction':r['mask_area_fraction'],
                                'edge_touch':r['mask_edge_touch'],
                                'xyz_points':r['xyz_points'], 'reasons':';'.join(r['reasons'])})
    return records, metrics


def summary_of(records, source):
    def counts(key): return {cam:sum(r[key]==cam for r in records) for cam in CAMERAS}
    return {'purpose':'Global/local observation role split; A/B with v0',
            'source_v0_jsonl':str(source), 'frame_count':len(records),
            'v0_selected_counts':counts('v0_selected'),
            'global_selected_counts':counts('global_selected'),
            'local_selected_counts':counts('local_selected'),
            'global_unavailable_frames':[r['frame'] for r in records if r['global_selected'] is None],
            'local_unavailable_frames':[r['frame'] for r in records if r['local_selected'] is None],
            'frames_global_differs_from_v0':[r['frame'] for r in records if r['global_selected'] != r['v0_selected']],
            'score_warning':'heuristics NOT calibrated to segmentation ground truth, garment fullness, task success or world-frame pose',
            'retained':['v1 deterministic pixel sampler','HSV blue v2 + yellow v3','per-camera optical XYZ'],
            'no_cross_camera_xyz_fusion':True}


def visualize(records, v0_by_frame, hdf5, calibrations, demo, output, frame_ids):
    """Diagnostic RGB + HSV mask sheets, no camera pose, no intrinsics needed."""
    import h5py
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from _common import get_camera_arrays, read_depth_frame
    from phase2_reorientation.perception import CameraFrame, MultiHSVRangeSegmenter, PerceptionProcessor
    bounds=[]
    for fp in calibrations:
        d=json.loads(fp.read_text(encoding='utf8'))
        bounds.append((tuple(d['hsv_lower']),tuple(d['hsv_upper'])))
    processor=PerceptionProcessor(segmenter=MultiHSVRangeSegmenter(
        hsv_ranges=tuple(bounds),min_component_pixels=80,keep_largest_components=3),min_mask_pixels=30)
    font=ImageFont.load_default()
    image_dir=output/'previews';image_dir.mkdir(exist_ok=True)
    selected={r['frame']:r for r in records if r['frame'] in frame_ids}
    with h5py.File(hdf5,'r') as f:
        obs=f[f'data/{demo}/obs']
        sources={c:get_camera_arrays(obs,c) for c in CAMERAS}
        for frame in sorted(selected):
            rec=selected[frame]
            sheet=Image.new('RGB',(5*256, 660),(245,247,250))
            dr=ImageDraw.Draw(sheet)
            dr.text((8,8),f"Frame {frame} | v0={rec['v0_selected']} | global={rec['global_selected']} | local={rec['local_selected']}",fill=(20,25,30),font=font)
            dr.text((8,25),'BLUE header = global selected; AMBER = local selected; gray = neither',fill=(20,25,30),font=font)
            rankings={r['camera']:r for r in rec['global_ranked']+rec['local_ranked']}
            for j,cam in enumerate(CAMERAS):
                x=j*256
                ar,ad,av=sources[cam]
                rgb=np.asarray(ar[frame],dtype=np.uint8)
                depth=read_depth_frame(ad,frame).astype(np.float32)
                valid=read_depth_frame(av,frame).astype(bool)
                detection=processor.process_frame(CameraFrame(camera_name=cam,rgb=rgb,depth_m=depth,depth_valid=valid))
                mask=np.asarray(detection.mask,dtype=bool)
                rank=rankings[cam]
                color=(24,87,153) if cam==rec['global_selected'] else ((180,112,21) if cam==rec['local_selected'] else (68,75,84))
                dr.rectangle((x+2,49,x+253,91),fill=color)
                dr.text((x+7,54),cam,font=font,fill='white')
                dr.text((x+7,70),f"score={rank['score']:.3f} mask={int(mask.sum())} {'OK' if rank['available'] else 'UNAVAILABLE'}",font=font,fill='white')
                rgb_img=Image.fromarray(rgb,'RGB')
                overlay=rgb.copy()
                overlay[mask]=np.clip(.65*overlay[mask]+.35*np.array([10,255,70]),0,255).astype(np.uint8)
                sheet.paste(rgb_img,(x,100))
                sheet.paste(Image.fromarray(overlay),(x,360))
                if int(mask.sum()) != int(next(i for i in v0_by_frame[frame]['cameras'] if i['camera']==cam)['mask_area_px']):
                    dr.text((x+3,621),'WARNING: mask mismatch!',fill=(180,0,0),font=font)
            sheet.save(image_dir/f'frame_{frame:06d}_v1_global_local.png',optimize=True)
    return len(selected)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v0-states',type=Path,required=True,help='Existing garment_observation_frames.jsonl')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hdf5',type=Path,help='Optional: render RGB/mask review images')
    p.add_argument('--calibration',type=Path,action='append',default=[])
    p.add_argument('--demo',default='demo_0')
    p.add_argument('--preview-frames',type=int,nargs='*',default=[1250,1500,1600,1700,1750,1850,1950,2050,2150,2200,2300,2400,2600])
    a=p.parse_args()
    if a.hdf5 and (not a.hdf5.is_file() or len(a.calibration)!=2):
        p.error('With --hdf5, supply existing HDF5 and exactly two --calibration JSONs')
    frames=[]
    for line in a.v0_states.read_text(encoding='utf8').splitlines():
        if line.strip():frames.append(json.loads(line))
    if not frames:p.error('v0 states file is empty')
    if len({int(f['frame']) for f in frames}) != len(frames):p.error('Duplicate frame IDs')
    records,metrics=prepare(frames)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'garment_observation_v1_frames.jsonl').open('w',encoding='utf8') as f:
        for record in records:f.write(json.dumps(record,allow_nan=False)+'\n')
    with (a.output/'garment_observation_v1_metrics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    report=summary_of(records,a.v0_states)
    if a.hdf5:
        source_by_frame={int(f['frame']):f for f in frames}
        count=visualize(records,source_by_frame,a.hdf5,a.calibration,a.demo,a.output,set(a.preview_frames))
        report['preview_frames_created']=count
    report['linux_maxrss_reported_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (a.output/'garment_observation_v1_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'frame_count':len(records),'global_counts':report['global_selected_counts'],
                      'local_counts':report['local_selected_counts'],'global_unavailable':report['global_unavailable_frames'],
                      'local_unavailable':report['local_unavailable_frames'],
                      'global_differs_from_v0':len(report['frames_global_differs_from_v0']),
                      'previews':report.get('preview_frames_created',0),
                      'peak_ram_mib':round(report['linux_maxrss_reported_kib']/1024,1)},indent=2))
    print('Saved:',a.output)


if __name__=='__main__':main()
