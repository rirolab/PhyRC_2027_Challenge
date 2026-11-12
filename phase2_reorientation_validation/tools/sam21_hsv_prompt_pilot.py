#!/usr/bin/env python3
"""SAM 2.1 tiny box-prompt pilot using HSV predictions as prompts (NOT ground truth).
Read-only benchmark; produces independent SAM predictions, visual comparisons, diagnostics.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def binary_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 127


def hsv_prompt(mask, pad):
    # One union bbox and center positive point from a high-distance mask pixel.
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return None
    h, w = mask.shape
    box = np.array([max(int(xx.min()) - pad, 0), max(int(yy.min()) - pad, 0),
                    min(int(xx.max()) + pad, w - 1), min(int(yy.max()) + pad, h - 1)], dtype=np.float32)
    # Reject boxes without meaningful pixel extent.
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    py, px = np.unravel_index(int(dist.argmax()), dist.shape)
    return box, np.array([[px, py]], dtype=np.float32)


def preview(rgb, hsv, sam, box, point, label, score):
    h, w = hsv.shape
    a = rgb.copy()
    b = rgb.copy()
    c = rgb.copy()
    b[hsv] = (0.45*b[hsv] + 0.55*np.array([0, 255, 110])).astype(np.uint8)
    c[sam] = (0.45*c[sam] + 0.55*np.array([250, 75, 210])).astype(np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(a, (x0, y0), (x1, y1), (255, 200, 0), 2)
        cv2.circle(a, tuple(point.astype(int).ravel()), 4, (255, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    tiles = []
    for title, im in [('RGB + HSV prompt',a),('HSV prediction',b),('SAM2.1 prediction',c)]:
        big = cv2.resize(im, (384, 384), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((420,384,3),dtype=np.uint8)
        canvas[:384]=big
        cv2.putText(canvas,title,(7,407),font,0.56,(240,240,240),1,cv2.LINE_AA)
        tiles.append(canvas)
    bottom = np.zeros((40,1152,3),dtype=np.uint8)
    cv2.putText(bottom,f'{label}   SAM model IoU score: {score:.3f} (NOT ground-truth IoU)',
                (12,27),font,0.55,(250,250,250),1,cv2.LINE_AA)
    return np.vstack((np.hstack(tiles),bottom))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/sam2.1/sam2.1_hiera_t.yaml')
    p.add_argument('--frames',nargs='+',type=int,default=[600,1250,1500,1600,1950,2200,2600])
    p.add_argument('--cameras',nargs='+',default=['robot_0_head','robot_1_head'])
    p.add_argument('--max-images',type=int,default=14)
    p.add_argument('--pad',type=int,default=8)
    args=p.parse_args()
    root=Path(args.benchmark).resolve()
    out=Path(args.output).resolve()
    (out/'masks').mkdir(parents=True,exist_ok=True)
    (out/'previews').mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check .venv-sam2 and GPU')
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    predictor=SAM2ImagePredictor(build_sam2(args.config,args.checkpoint,device='cuda'))
    records=[]
    with open(root/'annotation_manifest.csv',newline='') as f:
        manifest=list(csv.DictReader(f))
    chosen=[r for r in manifest if int(r['frame']) in args.frames and r['camera'] in args.cameras]
    chosen=sorted(chosen,key=lambda r:(int(r['frame']),r['camera']))[:args.max_images]
    torch.cuda.reset_peak_memory_stats()
    for row in chosen:
        sid=row['sample_id']
        im_bgr=cv2.imread(str(root/row['rgb']),cv2.IMREAD_COLOR)
        if im_bgr is None: raise FileNotFoundError(root/row['rgb'])
        rgb=cv2.cvtColor(im_bgr,cv2.COLOR_BGR2RGB)
        hsv=binary_mask(root/row['hsv_prediction'])
        prompt=hsv_prompt(hsv,args.pad)
        base={'sample_id':sid,'frame':int(row['frame']),'camera':row['camera'],
              'hsv_px':int(hsv.sum()),'sam_px':0,'overlap_iou_hsv_sam':None,
              'sam_predicted_iou':None,'infer_seconds':None,'status':'skipped_no_hsv_prompt'}
        if prompt is None:
            records.append(base)
            print(sid,'SKIP: empty HSV mask',flush=True)
            continue
        box,point=prompt
        t0=time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(rgb)
            masks,scores,_=predictor.predict(box=box,point_coords=point,
                                             point_labels=np.array([1],dtype=np.int32),
                                             multimask_output=False)
        torch.cuda.synchronize()
        duration=time.perf_counter()-t0
        sam=np.asarray(masks[0],dtype=bool)
        inter=int(np.logical_and(sam,hsv).sum())
        union=int(np.logical_or(sam,hsv).sum())
        overlap=inter/union if union else None
        score=float(scores[0])
        base.update(sam_px=int(sam.sum()),overlap_iou_hsv_sam=overlap,
                    sam_predicted_iou=score,infer_seconds=duration,status='ok')
        records.append(base)
        cv2.imwrite(str(out/'masks'/f'{sid}.png'),(sam.astype(np.uint8)*255))
        image=preview(rgb,hsv,sam,box,point,sid,score)
        cv2.imwrite(str(out/'previews'/f'{sid}.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        print(f'{sid} HSV={int(hsv.sum())} SAM={int(sam.sum())} '
              f'IoU(HSV,SAM)={overlap:.3f} sec={duration:.3f}',flush=True)
    cols=['sample_id','frame','camera','status','hsv_px','sam_px',
          'overlap_iou_hsv_sam','sam_predicted_iou','infer_seconds']
    with open(out/'pilot_metrics.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=cols); writer.writeheader();writer.writerows(records)
    summary={'purpose':'SAM2.1 HSV-prompted pilot, predictions are NOT GT',
             'checkpoint':str(args.checkpoint),'model_config':args.config,
             'total_samples':len(records),'succeeded':sum(r['status']=='ok' for r in records),
             'skipped_empty_hsv':sum(r['status']!='ok' for r in records),
             'gpu_peak_allocated_mib':round(torch.cuda.max_memory_allocated()/2**20,2),
             'gpu_peak_reserved_mib':round(torch.cuda.max_memory_reserved()/2**20,2),
             'warning':'IoU HSV/SAM is model agreement, NOT segmentation accuracy; HSV prompts bias SAM.'}
    with open(out/'pilot_summary.json','w') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
