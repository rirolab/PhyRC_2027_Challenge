#!/usr/bin/env python3
"""SAM2.1 garment segmentation v3: full-manifest A/B and guarded prompt refinement.

Only benchmark RGB/HSV predictions are read; no privileged simulator state.
Auto negative points are *hypotheses*, not mannequin labels. Saved auto candidates
are experimental. A selected result differs from baseline only under conservative
geometric acceptance checks; quality/accuracy requires manual GT verification.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from sam21_segmentation_v2 import (make_prompt, read_mask, get_negative_points,
    validate_negative_points, mask_diagnostics, tint, save_png)


def auto_negative_points(baseline, hsv, *, min_pixels=25, max_points=2, clearance=4):
    """Pick interior points from disconnected SAM islands lacking any HSV support.

    Such islands can be missed garment parts! Never treat as semantic evidence.
    Excludes islands touching HSV, and candidate points too close to HSV.
    """
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        baseline.astype(np.uint8), connectivity=8)
    # Distance from every pixel to the nearest HSV-positive pixel.
    dist_from_hsv = cv2.distanceTransform((~hsv).astype(np.uint8), cv2.DIST_L2, 3)
    pool = []
    for c in range(1, num):
        area = int(stats[c, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        component = lab == c
        if np.any(component & hsv):
            continue
        viable = component & (dist_from_hsv >= clearance)
        if not np.any(viable):
            continue
        # Deepest point in component, ensuring a reliable interior negative.
        center_dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 3)
        center_dist[~viable] = 0
        y, x = np.unravel_index(int(center_dist.argmax()), center_dist.shape)
        if center_dist[y, x] < 2:
            continue
        pool.append((area, float(x), float(y)))
    pool.sort(reverse=True)
    return np.asarray([[x, y] for _, x, y in pool[:max_points]],
                      dtype=np.float32).reshape(-1, 2)


def calc_metrics(mask, hsv, box, args):
    return mask_diagnostics(mask, hsv, box, args.min_main_component_fraction,
                            args.max_area_ratio, args.max_outside_bbox_fraction)


def guard_candidate(baseline, candidate, hsv, ba, ca, args):
    """A conservative *geometric* policy, not an accuracy estimator."""
    if not candidate.any():
        return False, 'candidate_empty'
    bh = int((baseline & hsv).sum())
    ch = int((candidate & hsv).sum())
    if bh > 0 and ch / bh < args.min_hsv_preservation:
        return False, 'lost_hsv_supported_pixels'
    bs = int(baseline.sum())
    cs = int(candidate.sum())
    if cs / max(bs, 1) < args.min_baseline_area_preservation:
        return False, 'large_mask_shrinkage'
    if cs / max(bs, 1) > args.max_growth:
        return False, 'large_mask_growth'
    bflags, cflags = set(ba['qa_flags']), set(ca['qa_flags'])
    if cflags - bflags:
        return False, 'new_qa_flags'
    bmain = ba['main_component_fraction']
    cmain = ca['main_component_fraction']
    if cmain + 0.005 < bmain:
        return False, 'worse_fragmentation'
    # No justified automatic switch for tiny/no changes.
    changed = int(np.logical_xor(baseline, candidate).sum())
    if changed / max(bs, 1) < args.min_changed_fraction:
        return False, 'negligible_change'
    bnoise = int((baseline & ~hsv).sum())
    cnoise = int((candidate & ~hsv).sum())
    if cnoise >= bnoise:
        return False, 'no_nonhsv_reduction'
    # Require a meaningful reduction in fragments OR clear removal of an
    # HSV-unsupported island; this is a proxy only, not proof of a correct mask.
    if cmain <= bmain + 0.01 and (bnoise - cnoise) / max(bs, 1) < 0.03:
        return False, 'insufficient_geometric_gain'
    return True, 'passed_geometric_guard_not_accuracy'


def overlay(rgb, mask, tint_rgb):
    return tint(rgb, mask, tint_rgb)


def make_preview(rgb, hsv, baseline, candidate, selected, box, positive, negatives, sid, reason):
    panels = [rgb.copy(), overlay(rgb, hsv, [30,230,130]),
              overlay(rgb, baseline, [230,60,200]),
              overlay(rgb, candidate, [230,160,35]),
              overlay(rgb, selected, [25,215,240])]
    x0,y0,x1,y1=np.round(box).astype(int)
    cv2.rectangle(panels[0],(x0,y0),(x1,y1),(255,220,0),1)
    cv2.circle(panels[0],tuple(np.round(positive[0]).astype(int)),3,(255,30,30),-1)
    for x,y in negatives:
        cv2.drawMarker(panels[0],(int(round(x)),int(round(y))),
                       (255,255,25),cv2.MARKER_TILTED_CROSS,8,1)
    names=['RGB + prompts','HSV (not GT)','SAM baseline','SAM candidate','Guarded output']
    outs=[]
    for frame,name in zip(panels,names):
        w=256
        panel=np.zeros((294,w,3),np.uint8)
        panel[:256]=frame
        cv2.putText(panel,name,(5,275),cv2.FONT_HERSHEY_SIMPLEX,.47,(235,235,235),1,cv2.LINE_AA)
        outs.append(panel)
    footer=np.zeros((44,1280,3),np.uint8)
    txt=f'{sid} | {reason}'
    cv2.putText(footer,txt[:165],(8,27),cv2.FONT_HERSHEY_SIMPLEX,.51,(235,235,235),1,cv2.LINE_AA)
    return np.vstack([np.hstack(outs),footer])


def load_samples(args):
    root=Path(args.benchmark).expanduser().resolve()
    with (root/'annotation_manifest.csv').open(newline='',encoding='utf-8-sig') as fp:
        manifest=list(csv.DictReader(fp))
    chosen=[]
    for row in manifest:
        if args.frames and int(row['frame']) not in args.frames:
            continue
        if args.cameras and row['camera'] not in args.cameras:
            continue
        chosen.append(row)
    chosen.sort(key=lambda r:(int(r['frame']),r['camera']))
    if args.max_images:
        chosen=chosen[:args.max_images]
    if not chosen:
        raise ValueError('No samples selected')
    manual=get_negative_points(Path(args.manual_negatives_csv) if args.manual_negatives_csv else None)
    unselected=set(manual)-set(r['sample_id'] for r in chosen)
    if unselected:
        raise ValueError(f'Manual negatives reference unselected sample IDs: {sorted(unselected)[:3]}')
    prepared=[]
    for row in chosen:
        sid=row['sample_id']
        raw=cv2.imread(str(root/row['rgb']),cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(str(root/row['rgb']))
        rgb=cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
        hsv=read_mask(root/row['hsv_prediction'])
        if rgb.shape[:2]!=hsv.shape:
            raise ValueError(f'RGB / HSV dimension mismatch: {sid}')
        if rgb.shape[:2] != (256,256):
            raise ValueError(f'Expected 256x256 preview for {sid}; got {rgb.shape[:2]}')
        manual_pts=validate_negative_points(manual.get(sid,[]),hsv)
        prepared.append((row,rgb,hsv,make_prompt(hsv,args.pad),manual_pts))
    return prepared


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--config',default='configs/sam2.1/sam2.1_hiera_t.yaml')
    ap.add_argument('--frames',nargs='*',type=int,default=None,help='Default ALL frames in manifest')
    ap.add_argument('--cameras',nargs='*',default=None,help='Default ALL 5 cameras')
    ap.add_argument('--max-images',type=int,default=None,help='Default ALL selected samples')
    ap.add_argument('--manual-negatives-csv',default=None)
    ap.add_argument('--no-auto-negatives',action='store_true')
    ap.add_argument('--auto-min-pixels',type=int,default=25)
    ap.add_argument('--max-auto-negatives',type=int,default=2)
    ap.add_argument('--auto-clearance',type=float,default=4)
    ap.add_argument('--pad',type=int,default=8)
    ap.add_argument('--min-main-component-fraction',type=float,default=.90)
    ap.add_argument('--max-area-ratio',type=float,default=1.40)
    ap.add_argument('--max-outside-bbox-fraction',type=float,default=.08)
    ap.add_argument('--min-hsv-preservation',type=float,default=.985)
    ap.add_argument('--min-baseline-area-preservation',type=float,default=.88)
    ap.add_argument('--max-growth',type=float,default=1.10)
    ap.add_argument('--min-changed-fraction',type=float,default=.005)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.max_images is not None and args.max_images < 1: ap.error('--max-images must be >= 1')
    if args.pad<0 or args.auto_min_pixels<1 or args.max_auto_negatives<0 or args.auto_clearance<0:
        ap.error('Invalid prompt settings')
    if not (0 < args.min_hsv_preservation <=1 and 0 < args.min_baseline_area_preservation <= 1 and args.max_growth >= 1):
        ap.error('Invalid preservation thresholds')
    out=Path(args.output).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output must be new/empty: {out}')
    prepared=load_samples(args)
    if args.dry_run:
        print(json.dumps({'dry_run':True,'samples':len(prepared),
            'with_hsv':sum(p is not None for _,_,_,p,_ in prepared),
            'with_manual_negatives':sum(len(n)>0 for *_,n in prepared),
            'note':'No model loaded and no output written'},indent=2))
        return
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in active environment')
    predictor=SAM2ImagePredictor(build_sam2(args.config,args.checkpoint,device='cuda'))
    out.mkdir(parents=True)
    for sub in ('baseline_masks','candidate_masks','selected_masks','previews'):
        (out/sub).mkdir()
    torch.cuda.reset_peak_memory_stats()
    records=[]
    for row,rgb,hsv,prompt,manual_neg in prepared:
        sid=row['sample_id']
        rec={'sample_id':sid,'frame':int(row['frame']),'camera':row['camera']}
        if prompt is None:
            rec.update(status='skipped_empty_hsv',selected='none',reason='no_HSV_prompt')
            records.append(rec)
            print(sid, 'SKIP empty HSV',flush=True)
            continue
        box,pos=prompt
        t0=time.perf_counter()
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            predictor.set_image(rgb)
            masks,scores,_=predictor.predict(box=box,point_coords=pos,
                 point_labels=np.array([1],np.int32),multimask_output=False)
            baseline=np.asarray(masks[0],bool)
            ba=calc_metrics(baseline,hsv,box,args)
            if len(manual_neg):
                negatives=manual_neg
                source='manual'
            elif args.no_auto_negatives:
                negatives=np.empty((0,2),np.float32)
                source='none'
            else:
                negatives=auto_negative_points(baseline,hsv,
                  min_pixels=args.auto_min_pixels,max_points=args.max_auto_negatives,
                  clearance=args.auto_clearance)
                source='automatic_geometric_hypothesis' if len(negatives) else 'none'
            if len(negatives):
                coords=np.concatenate([pos,negatives],axis=0)
                labels=np.array([1]+[0]*len(negatives),np.int32)
                masks2,scores2,_=predictor.predict(box=box,point_coords=coords,
                                       point_labels=labels,multimask_output=False)
                candidate=np.asarray(masks2[0],bool)
                candidate_score=float(scores2[0])
            else:
                candidate=baseline.copy()
                candidate_score=float(scores[0])
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-t0
        ca=calc_metrics(candidate,hsv,box,args)
        if len(negatives):
            accept,reason=guard_candidate(baseline,candidate,hsv,ba,ca,args)
        else:
            accept,reason=False,'no_negative_candidate'
        selected=candidate if accept else baseline
        selection='candidate' if accept else 'baseline'
        for directory,mask in [('baseline_masks',baseline),('candidate_masks',candidate),('selected_masks',selected)]:
            save_png(out/directory/(sid+'.png'),(mask.astype(np.uint8)*255))
        pic=make_preview(rgb,hsv,baseline,candidate,selected,box,pos,negatives,sid,reason)
        save_png(out/'previews'/(sid+'.png'),cv2.cvtColor(pic,cv2.COLOR_RGB2BGR))
        rec.update(status='ok',selected=selection,reason=reason,
                   negative_source=source,negative_points=negatives.tolist(),
                   baseline=ba,candidate=ca,
                   baseline_predicted_iou=float(scores[0]),candidate_predicted_iou=candidate_score,
                   seconds=round(elapsed,3),changed_fraction_vs_baseline=round(
                     float(np.logical_xor(baseline,candidate).sum())/max(int(baseline.sum()),1),5))
        records.append(rec)
        print(f'{sid}: {source}({len(negatives)}) selected={selection} '
              f'baseline={ba["qa_flags"]} candidate={ca["qa_flags"]} '
              f'reason={reason} sec={elapsed:.2f}',flush=True)
    (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    summary={'purpose':'SAM2.1 v3 full-manifest geometric prompt refinement, NOT validated accuracy',
      'source_benchmark':str(Path(args.benchmark).resolve()),
      'selected_samples':len(records),'succeeded':sum(r['status']=='ok' for r in records),
      'skipped_empty_hsv':sum(r['status']=='skipped_empty_hsv' for r in records),
      'automatic_prompt_samples':sum(r.get('negative_source')=='automatic_geometric_hypothesis' for r in records),
      'manual_prompt_samples':sum(r.get('negative_source')=='manual' for r in records),
      'accepted_candidate_samples':sum(r.get('selected')=='candidate' for r in records),
      'samples_requiring_review':[r['sample_id'] for r in records
           if r['status']=='ok' and (r['baseline']['qa_flags'] or r['candidate']['qa_flags'] or r.get('selected')=='candidate')],
      'peak_torch_allocated_mib':round(torch.cuda.max_memory_allocated()/1048576,2),
      'peak_torch_reserved_mib':round(torch.cuda.max_memory_reserved()/1048576,2),
      'safety_notes':['No semantic mannequin detector is present.',
         'Auto negatives target disconnected SAM islands with zero HSV overlap; they may be real garment.',
         'Selection guard is geometric only; it can still accept incorrect masks.',
         'Manual GT and independent episodes are required for accuracy and generalization claims.',
         'No weights trained; each frame is segmented independently, not tracked through motion.']}
    (out/'v3_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
