#!/usr/bin/env python3
"""PhyRC Automatic Semantic Landmark Proposals v2 (geometric, NOT trained).

SAM2.1 BASELINE mask only; sparse optical 3D via observed RGB-D only.
Silhouette rules propose tentative landmarks on likely isolated, uncropped T-shirts.
No automatic front/back inference; garment left/right is NOT resolvable from silhouette
alone. `left_*`/`right_*` are front-facing-view *hypotheses*, not certified semantics.
Ground truth annotations are used ONLY by evaluation, never by prediction.
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

KEYS=('collar_center','left_sleeve_tip','right_sleeve_tip','hem_left','hem_right')
HFOV={'overview':60.,'robot_0_wrist':70.,'robot_1_wrist':70.,'robot_0_head':70.,'robot_1_head':70.}


def write_json(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def read_manifest(root):
    with (root/'annotation_manifest.csv').open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def depth_project(xy,mask,depth,valid,camera,radius=2):
    x,y=map(int,xy);h,w=mask.shape
    if not(0<=x<w and 0<=y<h):return {'status':'invalid_pixel'}
    if not mask[y,x]:return {'status':'outside_sam_mask'}
    y0,y1=max(y-radius,0),min(y+radius+1,h)
    x0,x1=max(x-radius,0),min(x+radius+1,w)
    sl=(slice(y0,y1),slice(x0,x1))
    ok=mask[sl]&valid[sl]&np.isfinite(depth[sl])&(depth[sl]>0)
    if not ok.any():return {'status':'missing_local_depth'}
    z=float(np.median(depth[sl][ok]))
    fx=w/(2*math.tan(math.radians(HFOV[camera])/2))
    return {'status':'ok','xyz_optical_m':[round((x-w/2)*z/fx,6),round((y-h/2)*z/fx,6),round(z,6)],'local_valid_pixels':int(ok.sum()),'radius_px':radius}


def proposals(mask,min_area=250,border_max=0.10,shape_margin=0.12):
    """Conservative T-like silhouette heuristic. No learned semantic cues.

    PCA aligns the long silhouette axis, then cross-section max determines the
    likely shoulder-side vs hem-side. On ties/cropped/multiple islands, ABSTAIN.
    Rotate-invariant geometric side names A/B are mapped to camera horizontal
    for front-facing-view evaluation only (not wearable-side certainty).
    """
    h,w=mask.shape
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    if n<=1:return {},'empty_mask',{}
    sizes=stats[1:,cv2.CC_STAT_AREA]
    idx=int(np.argmax(sizes))+1
    major=(lab==idx)
    area=int(major.sum());all_area=int(mask.sum())
    meta={'mask_pixels':all_area,'main_component_pixels':area,'main_component_fraction':round(area/max(all_area,1),4)}
    if area<min_area:return {},'small_mask',meta
    if area/max(all_area,1)<0.87:return {},'fragmented_mask',meta
    # Cropped silhouette almost always breaks semantic extremity heuristics.
    band=np.zeros((h,w),bool);band[:3]=1;band[-3:]=1;band[:,:3]=1;band[:,-3:]=1
    border_ratio=float((major&band).sum())/area
    meta['border_fraction']=round(border_ratio,5)
    if border_ratio>border_max:return {},'cropped_mask',meta
    yy,xx=np.nonzero(major)
    xy=np.column_stack((xx.astype(float),yy.astype(float)))
    center=xy.mean(axis=0)
    cov=np.cov(xy.T)
    eigen,evec=np.linalg.eigh(cov);axis=evec[:,int(np.argmax(eigen))]
    # Stable orthogonal for consistent local geometry.
    side=np.array([-axis[1],axis[0]])
    long=(xy-center)@axis;cross=(xy-center)@side
    a,b=np.quantile(long,[.005,.995]);L=float(b-a)
    if L<16:return {},'short_silhouette',meta
    q=(long-a)/L
    def section(lo,hi):
        sel=(q>=lo)&(q<=hi)
        if np.count_nonzero(sel)<10:return None
        return float(np.quantile(cross[sel],.95)-np.quantile(cross[sel],.05))
    # Broad upper region (shoulder+sleeves) vs narrower lower hem.
    # Compare inner bands to suppress end-point contour anomalies.
    low=max([x for x in [section(.15,.33),section(.23,.42)] if x is not None],default=0)
    high=max([x for x in [section(.58,.77),section(.67,.85)] if x is not None],default=0)
    meta['band_widths_px']=[round(low,2),round(high,2)]
    if min(low,high)<5:return {},'insufficient_cross_section',meta
    contrast=abs(low-high)/max(low,high)
    meta['shoulder_width_contrast']=round(contrast,4)
    if contrast<shape_margin:return {},'shoulder_hem_ambiguous',meta
    # top = wider end; transform q so shoulder-side is at t=0
    top_at_small=low>high
    t=q if top_at_small else 1-q
    # for shoulder side, choose broadest section in 10%-46% of long length
    levels=np.arange(.12,.47,.04)
    widths=[]
    for level in levels:
        sel=np.abs(t-level)<=.045
        if np.count_nonzero(sel)<12:widths.append(0.)
        else:widths.append(float(np.quantile(cross[sel],.97)-np.quantile(cross[sel],.03)))
    peak=int(np.argmax(widths));t_sh=float(levels[peak]);w_sh=widths[peak]
    meta['shoulder_peak_t']=round(t_sh,3)
    if w_sh<12:return {},'no_clear_shoulder',meta
    ss=np.abs(t-t_sh)<=.085
    if np.count_nonzero(ss)<15:return {},'sparse_shoulder',meta
    left_s=xy[ss][int(np.argmin(cross[ss]))]
    right_s=xy[ss][int(np.argmax(cross[ss]))]
    # Collar center, shallow central top boundary of mask. An approximate
    # central upper contour, NOT a neckline detector.
    col_near=np.abs(cross-np.median(cross[ss]))<=max(5,w_sh*.13)
    if np.count_nonzero(col_near)<10:return {},'no_central_contour',meta
    colxy=xy[col_near][int(np.argmin(t[col_near]))]
    # hem corners from tail-most 15% with broad cross section
    hem_sel=t>=.82
    if np.count_nonzero(hem_sel)<12:return {},'no_hem_region',meta
    hem_a=xy[hem_sel][int(np.argmin(cross[hem_sel]))]
    hem_b=xy[hem_sel][int(np.argmax(cross[hem_sel]))]
    pts={'collar_center':colxy,'sleeve_a':left_s,'sleeve_b':right_s,'hem_a':hem_a,'hem_b':hem_b}
    # Make points actual on-mask; choose nearest mask pixel for rounded coordinates.
    def clamp(p):
        x=int(np.clip(round(float(p[0])),0,w-1));y=int(np.clip(round(float(p[1])),0,h-1))
        if major[y,x]:return [x,y]
        d=np.sum((xy-p)**2,axis=1);p2=xy[int(np.argmin(d))]
        return [int(p2[0]),int(p2[1])]
    pts={k:clamp(p) for k,p in pts.items()}
    # For pseudo-semantic hypothesis, use image-side horizontal ordering.
    # This fails for back views and camera rotations; always mark unverified.
    sa,sb=pts['sleeve_a'],pts['sleeve_b'];ha,hb=pts['hem_a'],pts['hem_b']
    # front-view wearer left appears to viewer's right for roughly upright view
    mapped={'collar_center':pts['collar_center'],
            'left_sleeve_tip':max([sa,sb],key=lambda z:z[0]),
            'right_sleeve_tip':min([sa,sb],key=lambda z:z[0]),
            'hem_left':max([ha,hb],key=lambda z:z[0]),
            'hem_right':min([ha,hb],key=lambda z:z[0])}
    if abs(sa[0]-sb[0])<10 or abs(ha[0]-hb[0])<8:
        return {},'side_mapping_ambiguous',meta
    return mapped,'proposed',meta


def evaluate(predictions,labels):
    pred={r['sample_id']:r for r in predictions}
    total=Counter();correct5=Counter();correct10=Counter();errors=[];pairs=[]
    unlabeled=[];status=Counter();per_key={k:{'gt':0,'pred':0,'pck5':0,'pck10':0,'distances':[]} for k in KEYS}
    samples_reviewed=0;points_reviewed=0;fully_empty=0
    for item in labels:
        if not item.get('reviewed',False):continue
        samples_reviewed+=1
        sid=item['sample_id'];points=item.get('points') or {}
        if not points:fully_empty+=1
        r=pred.get(sid,{'status':'no_prediction'});status[r['status']]+=1
        p=r.get('points_2d',{})
        for k,gtxy in points.items():
            if k not in per_key:continue
            total[k]+=1;points_reviewed+=1;per_key[k]['gt']+=1
            if k not in p:continue
            per_key[k]['pred']+=1
            d=float(np.linalg.norm(np.asarray(p[k])-np.asarray(gtxy)))
            errors.append(d);per_key[k]['distances'].append(d)
            if d<=5:correct5[k]+=1;per_key[k]['pck5']+=1
            if d<=10:correct10[k]+=1;per_key[k]['pck10']+=1
            pairs.append({'sample_id':sid,'key':k,'gt_xy':gtxy,'pred_xy':p[k],'pixel_error':round(d,3)})
    n=sum(total.values());covered=sum(x['pred'] for x in per_key.values())
    summary={'reviewed_images':samples_reviewed,'reviewed_images_without_landmark_labels':fully_empty,
     'total_annotated_landmark_points':n,'predicted_at_annotated_points':covered,
     'coverage_over_annotated_points':round(covered/max(n,1),5),
     'pck_at_5_over_all_annotated':round(sum(correct5.values())/max(n,1),5),
     'pck_at_10_over_all_annotated':round(sum(correct10.values())/max(n,1),5),
     'pck_at_5_conditional_on_prediction':round(sum(correct5.values())/covered,5) if covered else None,
     'pck_at_10_conditional_on_prediction':round(sum(correct10.values())/covered,5) if covered else None,
     'mean_pixel_error_on_predictions':round(float(np.mean(errors)),3) if errors else None,
     'median_pixel_error_on_predictions':round(float(np.median(errors)),3) if errors else None,
     'status_on_reviewed_images':dict(status),
     'per_key':{k:{**{q:v for q,v in d.items() if q!='distances'},'mean_pixel_error':round(float(np.mean(d['distances'])),3) if d['distances'] else None} for k,d in per_key.items()},
     'limitations':['19 labels are from one episode; evaluation is descriptive, NOT independent test generalization.',
      'Unannotated landmarks in partial images are UNKNOWN, not confirmed negatives.',
      'Silhouette cannot reliably determine garment front/back or wearer left/right.',
      'Left/right predictions assume roughly upright front-facing view; report as hypotheses only.',
      'GT annotations are never used to generate predictions.']}
    return summary,pairs


def run(args):
    bench=Path(args.benchmark);sam=Path(args.sam);out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    manifest=read_manifest(bench)
    records=[];status=Counter()
    for row in manifest:
        sid=row['sample_id'];cam=row['camera']
        f=sam/'baseline_masks'/f'{sid}.png'
        if not f.exists():
            records.append({'sample_id':sid,'camera':cam,'status':'no_sam_baseline','points_2d':{},'front_back':'unknown'});status['no_sam_baseline']+=1;continue
        raw=cv2.imread(str(f),cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape!=(256,256):raise ValueError('Invalid baseline mask '+str(f))
        mask=raw>127
        pts,decision,meta=proposals(mask,args.min_area,args.border_max,args.shape_margin)
        r={'sample_id':sid,'camera':cam,'status':decision,'points_2d':pts,
           'front_back':'unknown','front_back_source':'not_inferred',
           'left_right_semantics':'tentative_front_view_hypothesis' if pts else 'unavailable',
           'landmark_method':'pca_silhouette_cross_section_heuristic','diagnostics':meta,'landmarks_3d':{}}
        if pts and row.get('depth_valid'):
            path=bench/row['depth_valid']
            if path.exists():
                with np.load(path) as z:
                    depth=np.asarray(z['depth_m'],dtype=np.float32)
                    valid=np.asarray(z['depth_valid'],dtype=bool)
                if depth.shape!=mask.shape or valid.shape!=mask.shape:raise ValueError('Bad depth dimensions '+sid)
                r['landmarks_3d']={k:depth_project(v,mask,depth,valid,cam,args.radius) for k,v in pts.items()}
        records.append(r);status[decision]+=1
    write_json(out/'automatic_landmarks_v2.json',records)
    summ={'total_images':len(records),'status_counts':dict(status),
        'proposed_images':status['proposed'],'front_back_inferred':0,
        'notes':['SAM2.1 baseline_masks ONLY','2D contour heuristic; no learned semantic detection',
          'left/right is front-view *hypothesis*, cannot certify garment side from silhouette',
          '3D uses assumed camera HFOV from geometry v1; unverified intrinsics','3D success does not establish accuracy']}
    if args.labels:
        data=json.loads(Path(args.labels).read_text(encoding='utf-8'))
        if data.get('schema')!='phyrc-landmarks-visible-v1' or not isinstance(data.get('labels'),list):raise ValueError('Wrong labels schema')
        summ['evaluation'],pairs=evaluate(records,data['labels'])
        write_json(out/'evaluation_pairs_v2.json',pairs)
    write_json(out/'automatic_landmarks_summary_v2.json',summ)
    print(json.dumps(summ,indent=2,ensure_ascii=False))
    if args.previews:
        dest=out/'previews';dest.mkdir(exist_ok=True)
        by_id={r['sample_id']:r for r in records}
        for row in manifest:
            r=by_id[row['sample_id']]
            ip=bench/row['rgb']
            if not ip.exists():continue
            bgr=cv2.imread(str(ip));
            if bgr is None:continue
            colors=[(0,0,255),(0,220,0),(0,210,255),(255,150,30),(240,50,235)]
            for i,k in enumerate(KEYS):
                if k not in r['points_2d']:continue
                x,y=r['points_2d'][k];cv2.circle(bgr,(x,y),3,colors[i],-1)
                cv2.putText(bgr,k[:5],(min(x+3,210),max(y-3,12)),cv2.FONT_HERSHEY_SIMPLEX,.33,colors[i],1)
            cv2.putText(bgr,r['status'][:30],(3,249),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
            cv2.imwrite(str(dest/(row['sample_id']+'.png')),bgr)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark',required=True)
    ap.add_argument('--sam',required=True,help='SAM v3 folder, reads baseline_masks only')
    ap.add_argument('--labels',help='Human review JSON: evaluation ONLY')
    ap.add_argument('--output',required=True)
    ap.add_argument('--previews',action='store_true')
    ap.add_argument('--min-area',type=int,default=250)
    ap.add_argument('--border-max',type=float,default=.10)
    ap.add_argument('--shape-margin',type=float,default=.12)
    ap.add_argument('--radius',type=int,default=2)
    a=ap.parse_args()
    if not(0<=a.radius<=10):ap.error('--radius must be 0..10')
    if not(0<=a.border_max<=1):ap.error('--border-max 0..1')
    if not(0<=a.shape_margin<=1):ap.error('--shape-margin 0..1')
    run(a)
if __name__=='__main__':main()
