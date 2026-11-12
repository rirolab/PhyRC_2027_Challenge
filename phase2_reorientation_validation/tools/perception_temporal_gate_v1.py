#!/usr/bin/env python3
"""PhyRC Phase2: SAM2.1 Tiny temporal perception GO/NO-GO pilot.

RGB-D observations ONLY; fixed existing HSV prompt + SAM baseline, no negative points.
A 'PASS' requires HUMAN mask review; geometry-only metrics cannot prove accuracy.
Use `run` then edit review CSV then `assess` (without reloading SAM).
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import h5py
import numpy as np

CAMS=('overview','robot_0_head','robot_1_head','robot_0_wrist','robot_1_wrist')


def parse_windows(v):
    result=[]
    for s in v:
        parts=s.split(':')
        if len(parts)!=2: raise ValueError('window must be START:END (inclusive)')
        a,b=map(int,parts)
        if a<0 or b<a: raise ValueError(f'invalid window {s}')
        result.append((a,b))
    return result


def maskstats(mask):
    yy,xx=np.nonzero(mask)
    area=len(xx)
    if not area:return dict(area=0,area_fraction=0.,cx=None,cy=None,main_fraction=0.,bbox=None)
    n,_,sizes,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    main=int(sizes[1:,cv2.CC_STAT_AREA].max()) if n>1 else 0
    return dict(area=area,area_fraction=round(area/mask.size,6),
        cx=round(float(xx.mean()),2),cy=round(float(yy.mean()),2),
        main_fraction=round(main/area,5),bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())])


def prompt(hsv,pad=8):
    yy,xx=np.nonzero(hsv)
    if len(xx)==0:return None
    h,w=hsv.shape
    box=np.asarray([max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),
                    min(w-1,int(xx.max())+pad),min(h-1,int(yy.max())+pad)],dtype=np.float32)
    if box[2]<=box[0] or box[3]<=box[1]:return None
    distance=cv2.distanceTransform(hsv.astype('uint8'),cv2.DIST_L2,3)
    iy,ix=np.unravel_index(int(distance.argmax()),distance.shape)
    return box,np.asarray([[ix,iy]],dtype=np.float32)


def preview(rgb,hsv,mask,rec):
    a=rgb.copy();b=rgb.copy()
    a[hsv]=(.55*a[hsv]+.45*np.array([40,240,60])).astype('uint8')
    b[mask]=(.55*b[mask]+.45*np.array([255,60,185])).astype('uint8')
    out=np.hstack((rgb,a,b))
    bar=np.zeros((40,out.shape[1],3),dtype=np.uint8)
    text=f"{rec['sample_id']}  {rec['status']}  {rec['seconds']}s"
    cv2.putText(bar,text[:105],(6,24),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
    return np.vstack((out,bar))


def percentile(values,p):
    return round(float(np.percentile(values,p)),4) if values else None


def summarize(records,review_csv=None,thresholds=None):
    thresholds=thresholds or {}
    min_valid=float(thresholds.get('min_valid',.90))
    max_gap=int(thresholds.get('max_gap',3))
    min_review=int(thresholds.get('min_review',12))
    min_review_ok=float(thresholds.get('min_review_ok',.85))
    max_review_missed=int(thresholds.get('max_review_missed',0))
    max_p95_ms=float(thresholds.get('max_p95_ms',250))
    max_peak_gb=float(thresholds.get('max_peak_gb',2.0))
    counts=Counter(r['status'] for r in records)
    total=len(records); valid=counts['ok']
    gap=0;max_streak=0
    for key in sorted(set((r['window'],r['camera']) for r in records)):
        seq=sorted((r for r in records if (r['window'],r['camera'])==key),key=lambda r:r['frame'])
        gap=0
        for r in seq:
            gap=(gap+1) if r['status']!='ok' else 0
            max_streak=max(max_streak,gap)
    times=[r['seconds']*1000 for r in records if r.get('seconds') is not None]
    peak=max([r.get('peak_gpu_gb',0) for r in records] or [0])
    # Stepwise diagnostics only: changed camera viewpoints invalidate rigid overlap assumptions.
    jump=[]
    for key in sorted(set((r['window'],r['camera']) for r in records)):
        seq=sorted((r for r in records if (r['window'],r['camera'])==key),key=lambda r:r['frame'])
        for prev,curr in zip(seq,seq[1:]):
            a,b=prev['mask'],curr['mask']
            if a['area'] and b['area']:
                area_ratio=max(a['area'],b['area'])/max(1,min(a['area'],b['area']))
                shift=np.hypot(a['cx']-b['cx'],a['cy']-b['cy'])/np.hypot(256,256)
                if area_ratio>2.5 or shift>0.30:
                    jump.append({'from':prev['sample_id'],'to':curr['sample_id'],
                                'area_ratio':round(area_ratio,2),'centroid_shift_diag':round(float(shift),3)})
    details={'evaluated_samples':total,'status_counts':dict(counts),'valid_fraction':round(valid/max(1,total),4),
            'longest_invalid_streak_sampled_frames':max_streak,'latency_p50_ms':percentile(times,50),
            'latency_p95_ms':percentile(times,95),'gpu_peak_allocated_gb':round(peak,3),
            'jump_warnings_count':len(jump),'jump_warnings_examples':jump[:15],
            'thresholds':dict(min_valid=min_valid,max_gap=max_gap,min_review=min_review,
               min_review_ok=min_review_ok,max_review_missed=max_review_missed,
               max_p95_ms=max_p95_ms,max_peak_gb=max_peak_gb)}
    metrics_ok=(total>=20 and valid/max(1,total)>=min_valid and max_streak<=max_gap and
        (details['latency_p95_ms'] is not None and details['latency_p95_ms']<=max_p95_ms) and peak<=max_peak_gb)
    reviews=[]
    if review_csv and Path(review_csv).is_file():
        with open(review_csv,newline='',encoding='utf-8-sig') as f: reviews=list(csv.DictReader(f))
    valid_ids={r['sample_id'] for r in records}
    yes=[]; visible_misses=[];complete=[]
    for row in reviews:
        if row.get('sample_id') not in valid_ids:continue
        vis=row.get('garment_visible','').strip().lower()
        ok=row.get('mask_acceptable','').strip().lower()
        if vis in ('0','1') and ok in ('0','1'):
            complete.append(row)
            if vis=='1':
                yes.append(row)
                if ok=='0' and next(r for r in records if r['sample_id']==row['sample_id'])['status']!='ok':
                    visible_misses.append(row['sample_id'])
    ok_review=sum(r['mask_acceptable'].strip()=='1' for r in yes)
    review_rate=ok_review/len(yes) if yes else None
    review_ready=len(complete)>=min_review and len(yes)>=max(6,min_review//2)
    review_pass=review_ready and review_rate>=min_review_ok and len(visible_misses)<=max_review_missed
    details['human_review']={'complete':len(complete),'visible':len(yes),
       'acceptable_visible':ok_review,'acceptable_rate_visible':round(review_rate,4) if review_rate is not None else None,
       'visible_missing_or_invalid':visible_misses,'review_ready':review_ready}
    if not review_ready: decision='INSUFFICIENT_REVIEW'
    else: decision='PASS_PILOT' if (metrics_ok and review_pass) else 'FAIL_PILOT'
    details['decision']=decision
    details['machine_criteria_pass']=bool(metrics_ok)
    details['scope']='Pilot-only: no proof of generalization to unseen episodes or closed-loop safety'
    details['interpretation']='Temporal jumps are warnings, not automatically errors; masks can change as camera and cloth move.'
    return details


def read_params(args):
    return dict(min_valid=args.min_valid,max_gap=args.max_gap,min_review=args.min_review,
                min_review_ok=args.min_review_ok,max_review_missed=args.max_review_missed,
                max_p95_ms=args.max_p95_ms,max_peak_gb=args.max_peak_gb)


def add_gates(ap):
    ap.add_argument('--min-valid',type=float,default=.90)
    ap.add_argument('--max-gap',type=int,default=3)
    ap.add_argument('--min-review',type=int,default=12)
    ap.add_argument('--min-review-ok',type=float,default=.85)
    ap.add_argument('--max-review-missed',type=int,default=0)
    ap.add_argument('--max-p95-ms',type=float,default=250)
    ap.add_argument('--max-peak-gb',type=float,default=2.0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run',help='Run existing HSV+SAM baseline on consecutive HDF5 observations')
    run.add_argument('--hdf5',type=Path,required=True)
    run.add_argument('--demo',default='demo_0')
    run.add_argument('--calibration',action='append',type=Path,required=True,
                     help='Repeat for each original HSV calibration JSON (same as benchmark)')
    run.add_argument('--checkpoint',type=Path,default=Path.home()/'sam2/checkpoints/sam2.1_hiera_tiny.pt')
    run.add_argument('--config',default='configs/sam2.1/sam2.1_hiera_t.yaml')
    run.add_argument('--window',action='append',default=None,help='START:END inclusive; repeat for distinct manipulation phases')
    run.add_argument('--step',type=int,default=5,help='Sampling interval in recorded frames (default 5 = 4 Hz at 20 Hz)')
    run.add_argument('--camera',action='append',choices=CAMS,default=None)
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--min-pixels',type=int,default=80)
    run.add_argument('--pad',type=int,default=8)
    run.add_argument('--review-count',type=int,default=16)
    add_gates(run)
    assess=sub.add_parser('assess',help='Re-assess saved records after completing human review CSV; NO GPU needed')
    assess.add_argument('--output',type=Path,required=True)
    add_gates(assess)
    args=ap.parse_args()
    if args.command=='assess':
        out=args.output
        records=json.loads((out/'frame_results.json').read_text(encoding='utf8'))
        result=summarize(records,out/'review_sheet.csv',read_params(args))
        (out/'gate_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(result,indent=2));return
    windows=parse_windows(args.window or ['950:1050','1700:1800','2350:2450'])
    if args.step<1 or args.review_count<1:ap.error('step and review-count must be positive')
    out=args.output
    if out.exists() and any(out.iterdir()):ap.error('output must be new or empty; preserve past runs')
    for p in args.calibration:
        if not p.is_file():ap.error(f'missing calibration: {p}')
    if not args.checkpoint.is_file():ap.error(f'missing checkpoint: {args.checkpoint}')
    # Match the HDF5 benchmark extractor exactly, rather than reimplementing HSV morphology.
    root=Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from phase2_reorientation.perception import CameraFrame,PerceptionProcessor,MultiHSVRangeSegmenter
    bounds=[]
    for p in args.calibration:
        d=json.loads(p.read_text(encoding='utf8'))
        bounds.append((tuple(d['hsv_lower']),tuple(d['hsv_upper'])))
    processor=PerceptionProcessor(segmenter=MultiHSVRangeSegmenter(hsv_ranges=tuple(bounds),
             min_component_pixels=80,keep_largest_components=3),min_mask_pixels=30)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for inference pilot')
    model=SAM2ImagePredictor(build_sam2(args.config,str(args.checkpoint),device='cuda'))
    from _common import get_camera_arrays,read_depth_frame
    out.mkdir(parents=True,exist_ok=True);(out/'previews').mkdir()
    cams=args.camera or ['robot_0_head','robot_1_head']
    cams=list(dict.fromkeys(cams))
    rows=[]
    with h5py.File(args.hdf5,'r') as f:
        obs=f[f'data/{args.demo}/obs']
        sources={cam:get_camera_arrays(obs,cam) for cam in cams}
        num=len(next(iter(sources.values()))[0])
        for wi,(a,b) in enumerate(windows):
            if b>=num:raise ValueError(f'Window {a}:{b} exceeds available 0..{num-1}')
            for frame in range(a,b+1,args.step):
                for cam in cams:
                    rgb_ds,dep_ds,valid_ds=sources[cam]
                    rgb=np.asarray(rgb_ds[frame],dtype=np.uint8)
                    dep=np.asarray(read_depth_frame(dep_ds,frame),dtype=np.float32)
                    dep_valid=np.asarray(read_depth_frame(valid_ds,frame),dtype=bool)
                    hsv=np.asarray(processor.process_frame(CameraFrame(camera_name=cam,rgb=rgb,
                                     depth_m=dep,depth_valid=dep_valid)).mask,dtype=bool)
                    sid=f'{args.demo}_{frame:06d}_{cam}'
                    pr=prompt(hsv,args.pad)
                    mask=np.zeros_like(hsv)
                    elapsed=None
                    if pr is None:status='no_hsv_prompt'
                    else:
                        box,pos=pr
                        torch.cuda.synchronize();start=time.perf_counter()
                        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                            model.set_image(rgb)
                            masks,_,_=model.predict(box=box,point_coords=pos,
                                 point_labels=np.array([1],dtype=np.int32),multimask_output=False)
                        torch.cuda.synchronize()
                        elapsed=round(time.perf_counter()-start,4)
                        mask=np.asarray(masks[0],dtype=bool)
                        st=maskstats(mask)
                        status='ok' if st['area']>=args.min_pixels and st['main_fraction']>=.85 and st['area_fraction']<=.65 else 'mask_quality_flag'
                    stats=maskstats(mask)
                    record={'sample_id':sid,'window':wi,'frame':frame,'camera':cam,
                            'status':status,'seconds':elapsed,'hsv':maskstats(hsv),'mask':stats,
                            'peak_gpu_gb':round(torch.cuda.max_memory_allocated()/1024**3,4)}
                    rows.append(record)
                    # Review selection made AFTER recording, so no preview flood.
                    (out/'previews'/(sid+'.png')).write_bytes(cv2.imencode('.png',cv2.cvtColor(preview(rgb,hsv,mask,record),cv2.COLOR_RGB2BGR))[1].tobytes())
                    print(f'{sid}: {status} mask={stats["area"]} runtime={elapsed}',flush=True)
    (out/'frame_results.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    # Balanced sample of all camera+window strata; deterministic even when some strata are short.
    strata=defaultdict(list)
    for r in rows:strata[(r['window'],r['camera'])].append(r)
    chosen=[];seen=set()
    keys=sorted(strata)
    quota=max(1,args.review_count//len(keys))
    for key in keys:
        seq=strata[key]
        for i in np.linspace(0,len(seq)-1,min(quota,len(seq)),dtype=int):
            rec=seq[int(i)]
            if rec['sample_id'] not in seen:chosen.append(rec);seen.add(rec['sample_id'])
    for rec in rows:
        if len(chosen)>=args.review_count:break
        if rec['sample_id'] not in seen:chosen.append(rec);seen.add(rec['sample_id'])
    with (out/'review_sheet.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f)
        writer.writerow(['sample_id','preview_path','garment_visible','mask_acceptable','notes'])
        for r in chosen[:args.review_count]:
            writer.writerow([r['sample_id'],'previews/'+r['sample_id']+'.png','','',''])
    summary=summarize(rows,out/'review_sheet.csv',read_params(args))
    (out/'gate_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('\nGATE:',summary['decision'],'\nMachine-only criteria:',summary['machine_criteria_pass'])
    print('REVIEW REQUIRED: inspect preview panels (RGB | HSV | SAM), fill garment_visible and mask_acceptable with 0/1 in review_sheet.csv, then run assess.')

if __name__=='__main__':main()
