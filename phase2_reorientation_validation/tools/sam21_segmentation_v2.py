#!/usr/bin/env python3
"""PhyRC SAM2.1 segmentation v2: controlled negative-prompt experiments.

Reads existing benchmark RGB+HSV and does NOT change baseline or ground-truth data.
Negative points are human-provided image coordinates (not automatically inferred
from HSV); both original and negative-prompt SAM masks are retained for review.
Mask QA flags are diagnostics, not proof of correctness or automatic clipping.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FRAMES = [600, 1250, 1500, 1600, 1950, 2200, 2600]
DEFAULT_CAMS = ['robot_0_head', 'robot_1_head']


def read_mask(path: Path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(str(path))
    return a > 127


def make_prompt(mask: np.ndarray, pad: int):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    box = np.array([max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
                    min(w-1, int(xs.max())+pad), min(h-1, int(ys.max())+pad)], dtype=np.float32)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    dist = cv2.distanceTransform(mask.astype('uint8'), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(dist.argmax()), dist.shape)
    return box, np.array([[x, y]], dtype=np.float32)


def get_negative_points(path: Path | None):
    """CSV rows sample_id,x,y. Empty path means no negative prompts."""
    lookup = {}
    if path is None:
        return lookup
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open(newline='', encoding='utf-8-sig') as f:
        rdr = csv.DictReader(f)
        if not {'sample_id', 'x', 'y'}.issubset(rdr.fieldnames or []):
            raise ValueError('negative CSV requires columns sample_id,x,y')
        for row in rdr:
            if not (row['sample_id'].strip() and row['x'].strip() and row['y'].strip()):
                continue
            sid = row['sample_id'].strip()
            xy = (float(row['x']), float(row['y']))
            if not all(np.isfinite(v) for v in xy):
                raise ValueError('Non-finite negative point for '+sid)
            lookup.setdefault(sid, []).append(xy)
    return lookup


def validate_negative_points(pts, mask):
    h, w = mask.shape
    out = []
    for x, y in pts:
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            raise ValueError(f'Negative point ({x},{y}) out of {w}x{h} image')
        if mask[iy, ix]:
            raise ValueError(f'Negative point ({x},{y}) lies INSIDE HSV prediction; review annotation')
        out.append([float(x), float(y)])
    return np.asarray(out, dtype=np.float32).reshape(-1, 2)


def mask_diagnostics(sam: np.ndarray, hsv: np.ndarray, box: np.ndarray, min_main: float,
                     max_area_ratio: float, max_outside_box: float):
    a = np.uint8(sam)
    n = int(a.sum())
    h = int(hsv.sum())
    intersection = int(np.logical_and(sam, hsv).sum())
    union = int(np.logical_or(sam, hsv).sum())
    components, labels, stats, _ = cv2.connectedComponentsWithStats(a, 8)
    sizes = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, components)]
    biggest = max(sizes, default=0)
    x0, y0, x1, y1 = np.round(box).astype(int)
    within = np.zeros_like(sam)
    within[y0:y1+1, x0:x1+1] = True
    outside = int(np.logical_and(sam, ~within).sum())
    ratio = n / max(h, 1)
    outside_frac = outside / max(n, 1)
    largest_frac = biggest / max(n, 1)
    flags = []
    if n == 0: flags.append('empty_sam')
    if n and largest_frac < min_main: flags.append('fragmented')
    if n and ratio > max_area_ratio: flags.append('area_expansion')
    if n and outside_frac > max_outside_box: flags.append('outside_prompt_bbox')
    return dict(sam_px=n, hsv_px=h,
                agreement_iou=round(intersection/union, 5) if union else None,
                sam_to_hsv_area_ratio=round(ratio, 5),
                main_component_fraction=round(largest_frac, 5),
                outside_bbox_fraction=round(outside_frac, 5),
                qa_flags=flags)


def tint(rgb, mask, color):
    out = rgb.copy()
    out[mask] = np.clip(0.48*out[mask].astype(float)+0.52*np.array(color), 0, 255).astype(np.uint8)
    return out


def preview(rgb, hsv, baseline, candidate, box, pos, neg, title):
    frames = [rgb.copy(), tint(rgb, hsv, [30, 230, 130]),
              tint(rgb, baseline, [230, 60, 200]),
              tint(rgb, candidate, [230, 160, 35])]
    x0, y0, x1, y1 = np.round(box).astype(int)
    cv2.rectangle(frames[0], (x0,y0), (x1,y1), (255,220,0), 2)
    cv2.circle(frames[0], tuple(np.round(pos[0]).astype(int)), 4, (255,50,50), -1)
    for x,y in neg:
        cv2.drawMarker(frames[0], (int(round(x)),int(round(y))), (255,255,30),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
    names = ['RGB: pos(red), neg(yellow)', 'HSV prediction', 'SAM baseline', 'SAM with negatives' if len(neg) else 'SAM baseline (no negatives)']
    panels = []
    for im, name in zip(frames, names):
        big = cv2.resize(im, (384,384), interpolation=cv2.INTER_NEAREST)
        panel = np.zeros((423,384,3), dtype=np.uint8)
        panel[:384] = big
        cv2.putText(panel, name, (8,408), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235,235,235), 1, cv2.LINE_AA)
        panels.append(panel)
    footer = np.zeros((38,1536,3), dtype=np.uint8)
    cv2.putText(footer,title,(10,26),cv2.FONT_HERSHEY_SIMPLEX,0.55,(235,235,235),1,cv2.LINE_AA)
    return np.vstack([np.hstack(panels),footer])


def save_png(path, array):
    if not cv2.imwrite(str(path), array):
        raise OSError(f'Could not write {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--config', default='configs/sam2.1/sam2.1_hiera_t.yaml')
    ap.add_argument('--negatives-csv', default=None,
                    help='Optional manual negative points CSV: sample_id,x,y. Never inferred automatically.')
    ap.add_argument('--frames', nargs='+', type=int, default=DEFAULT_FRAMES)
    ap.add_argument('--cameras', nargs='+', default=DEFAULT_CAMS)
    ap.add_argument('--max-images', type=int, default=14)
    ap.add_argument('--pad', type=int, default=8)
    ap.add_argument('--min-main-component-fraction', type=float, default=0.90)
    ap.add_argument('--max-area-ratio', type=float, default=1.40)
    ap.add_argument('--max-outside-bbox-fraction', type=float, default=0.08)
    ap.add_argument('--dry-run', action='store_true', help='Check inputs/annotations without loading model')
    args = ap.parse_args()
    root = Path(args.benchmark).resolve()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output must be empty/new to preserve comparisons: {out}')
    if args.max_images <= 0 or args.pad < 0:
        raise ValueError('max-images must be >0, pad >=0')
    if not (0 < args.min_main_component_fraction <= 1 and args.max_area_ratio > 0 and 0 <= args.max_outside_bbox_fraction <= 1):
        raise ValueError('Invalid QA threshold')
    neg_lookup = get_negative_points(Path(args.negatives_csv) if args.negatives_csv else None)
    with (root/'annotation_manifest.csv').open(newline='', encoding='utf-8-sig') as f:
        manifest = list(csv.DictReader(f))
    chosen = sorted((r for r in manifest if int(r['frame']) in args.frames and r['camera'] in args.cameras),
                    key=lambda r:(int(r['frame']),r['camera']))[:args.max_images]
    if not chosen: raise ValueError('No matching rows in manifest')
    known = set(r['sample_id'] for r in chosen)
    extra = sorted(set(neg_lookup)-known)
    if extra:
        raise ValueError(f'Negative CSV IDs not in selected samples: {extra[:5]}')
    # Preflight all chosen samples and point annotations before costly model initialization.
    prepared = []
    for r in chosen:
        sid = r['sample_id']
        rgbpath = root / r['rgb']
        im = cv2.imread(str(rgbpath), cv2.IMREAD_COLOR)
        if im is None: raise FileNotFoundError(str(rgbpath))
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        hsv = read_mask(root/r['hsv_prediction'])
        if hsv.shape != rgb.shape[:2]: raise ValueError(f'RGB/mask size mismatch: {sid}')
        prompt = make_prompt(hsv, args.pad)
        neg = validate_negative_points(neg_lookup.get(sid, []), hsv)
        prepared.append((r, rgb, hsv, prompt, neg))
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, samples=len(prepared),
                              samples_with_hsv=sum(p is not None for _,_,_,p,_ in prepared),
                              annotated_negatives=sum(len(n) for *_,n in prepared),
                              note='No SAM model loaded, no masks generated'), indent=2))
        return
    # Lazy model import keeps annotation preflight runnable without CUDA.
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in SAM environment')
    out.mkdir(parents=True, exist_ok=True)
    for folder in ('baseline_masks','negative_prompt_masks','previews'):
        (out/folder).mkdir()
    predictor = SAM2ImagePredictor(build_sam2(args.config, args.checkpoint, device='cuda'))
    torch.cuda.reset_peak_memory_stats()
    records = []
    for row, rgb, hsv, prompt, neg in prepared:
        sid = row['sample_id']
        rec = dict(sample_id=sid,frame=int(row['frame']),camera=row['camera'],negative_points=int(len(neg)),status='skipped_empty_hsv')
        if prompt is None:
            records.append(rec); print(sid,'SKIP empty HSV',flush=True); continue
        box, pos = prompt
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(box=box, point_coords=pos,
                         point_labels=np.array([1],dtype=np.int32), multimask_output=False)
            baseline = np.asarray(masks[0], dtype=bool)
            baseline_score = float(scores[0])
            if len(neg):
                pts = np.concatenate([pos,neg],axis=0)
                labels = np.array([1]+[0]*len(neg),dtype=np.int32)
                masks2, scores2, _ = predictor.predict(box=box, point_coords=pts,
                                            point_labels=labels,multimask_output=False)
                candidate = np.asarray(masks2[0],dtype=bool)
                candidate_score = float(scores2[0])
            else:
                candidate = baseline.copy()
                candidate_score = baseline_score
        torch.cuda.synchronize()
        sec = time.perf_counter()-t0
        ba = mask_diagnostics(baseline,hsv,box,args.min_main_component_fraction,
                              args.max_area_ratio,args.max_outside_bbox_fraction)
        ca = mask_diagnostics(candidate,hsv,box,args.min_main_component_fraction,
                              args.max_area_ratio,args.max_outside_bbox_fraction)
        # Auto-switching by HSV overlap is explicitly avoided: agreement != accuracy.
        both_union = int(np.logical_or(baseline,candidate).sum())
        both_iou = (int(np.logical_and(baseline,candidate).sum())/both_union) if both_union else None
        rec.update(status='ok',seconds=round(sec,4),baseline_predicted_iou=round(baseline_score,5),
                   negative_predicted_iou=round(candidate_score,5),
                   baseline=ba,with_negative=ca,
                   baseline_vs_negative_iou=round(both_iou,5) if both_iou is not None else None)
        records.append(rec)
        save_png(out/'baseline_masks'/f'{sid}.png',np.uint8(baseline)*255)
        save_png(out/'negative_prompt_masks'/f'{sid}.png',np.uint8(candidate)*255)
        pic = preview(rgb,hsv,baseline,candidate,box,pos,neg,sid)
        save_png(out/'previews'/f'{sid}.png',cv2.cvtColor(pic,cv2.COLOR_RGB2BGR))
        print(f'{sid}: baseline_flags={ba["qa_flags"]} negative_flags={ca["qa_flags"]} negatives={len(neg)} sec={sec:.2f}',flush=True)
    with (out/'results.json').open('w') as f: json.dump(records,f,indent=2)
    summary = dict(purpose='SAM2.1 controlled negative prompt A/B, predictions NOT ground truth',
                   checkpoint=args.checkpoint, config=args.config, total_samples=len(records),
                   succeeded=sum(r['status']=='ok' for r in records),
                   samples_with_manual_negatives=sum(r.get('negative_points',0)>0 for r in records),
                   gpu_peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1048576,2),
                   gpu_peak_reserved_mib=round(torch.cuda.max_memory_reserved()/1048576,2),
                   counts_of_flags={name:sum(name in r.get('with_negative',{}).get('qa_flags',[]) for r in records)
                                    for name in ('empty_sam','fragmented','area_expansion','outside_prompt_bbox')},
                   caveats=['QA flags are heuristic diagnostics, not accuracy measures.',
                            'Negative prompt pixels must be manually verified as non-garment.',
                            'Identical baseline and candidate when no negatives supplied.',
                            'Never treat predicted SAM IoU/HSV agreement as GT IoU.',
                            'One demo does not constitute a held-out evaluation.'])
    with (out/'v2_summary.json').open('w') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
