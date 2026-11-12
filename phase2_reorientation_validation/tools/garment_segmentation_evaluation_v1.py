#!/usr/bin/env python3
"""PhyRC Garment Segmentation Evaluation v1: independent RGB-only human GT and fair mask metrics.

Commands:
 prepare: creates a self-contained offline polygon annotation HTML with ONLY original RGB.
 evaluate: rasterizes independently annotated polygons and compares HSV, SAM baseline, SAM v3 selected.

No GPU, simulator privileged values, or SAM model required. Requires numpy, cv2.
"""
from __future__ import annotations
import argparse
import base64
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

SCHEMA = 'phyrc-garment-visible-mask-v1'
CAMERAS = ('overview','robot_0_head','robot_1_head','robot_0_wrist','robot_1_wrist')

def manifest_read(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(not r.get('sample_id') or not r.get('rgb') for r in rows):
        raise ValueError('Missing/invalid benchmark annotation_manifest.csv')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample ID in manifest')
    return rows


def select_samples(rows, result_list, count):
    by_id = {r['sample_id']: r for r in rows}
    selected = []
    def take(sid):
        if sid in by_id and sid not in selected and len(selected) < count:
            selected.append(sid)
    # Coverage of the five accepted changes and the six HSV-empty failures is important.
    for rec in result_list:
        if rec.get('selected') == 'candidate': take(rec['sample_id'])
    for rec in result_list:
        if rec.get('status') == 'skipped_empty_hsv': take(rec['sample_id'])
    # Spread remaining samples across frames and views, not only the end of one episode.
    ordered = sorted(rows, key=lambda r: (int(r['frame']), CAMERAS.index(r['camera']) if r['camera'] in CAMERAS else 9))
    frames = sorted({int(r['frame']) for r in ordered})
    cams = [c for c in CAMERAS if any(r['camera'] == c for r in ordered)]
    for fi,frame in enumerate(frames):
        # Rotate camera priority with time to avoid a mostly-overview annotation set.
        for offset in range(len(cams)):
            camera = cams[(fi+offset) % len(cams)]
            for r in ordered:
                if int(r['frame']) == frame and r['camera'] == camera:
                    take(r['sample_id'])
    return [by_id[i] for i in selected]


HTML = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhyRC – Human Ground Truth Annotation (RGB only)</title>
<style>body{font-family:system-ui,Arial,sans-serif;background:#111827;color:#f3f4f6;margin:22px;max-width:970px}h2{margin:0 0 12px}button,select,input{font-size:15px;padding:9px;border-radius:6px;border:1px solid #8b95a5;background:#253146;color:white;margin:4px}button:hover{background:#33465c}button:disabled{opacity:.4}canvas{width:512px;height:512px;image-rendering:pixelated;cursor:crosshair;border:1px solid #9ca3af;background:#111;touch-action:none} .layout{display:flex;gap:20px;flex-wrap:wrap}.controls{width:340px}.note{color:#cbd5e1;font-size:13px;line-height:1.6}.status{color:#a7f3d0}.danger{color:#fca5a5}.progress{margin:15px 0}</style>
</head><body><h2>PhyRC · Garment GT (visible pixels)</h2>
<p class="note">Chỉ hiển thị RGB gốc; KHÔNG xem HSV/SAM khi vẽ để tránh thiên lệch. Click lần lượt quanh đường viền áo NHÌN THẤY. Nhấn “Đóng polygon” để thêm vùng. Chọn “Xóa lỗ” cho vùng đầu mannequin/nền nằm trong polygon. Không gán nhãn những vùng áo bị che khuất.</p>
<div class="progress" id="header"></div><div class="layout"><div><canvas id="image" width="256" height="256"></canvas><div class="note">Ảnh nguồn 256×256; click được ánh xạ đúng pixel. Polygon xanh = áo; polygon cam = vùng loại trừ.</div></div>
<div class="controls">
<div><button onclick="prev()">← Trước</button><button onclick="next()">Tiếp →</button></div>
<div><select id="mode"><option value="add">Vẽ vùng áo (+)</option><option value="erase">Xóa lỗ (−)</option></select><button onclick="finish()">Đóng polygon</button></div>
<div><button onclick="undoPoint()">Lùi một điểm</button><button onclick="undoPolygon()">Xóa polygon cuối</button></div>
<div><button onclick="mark('labeled')">Xác nhận nhãn áo</button><button onclick="mark('empty')">Ảnh không có áo</button></div>
<div><button onclick="mark('unobservable')">Không xác định được</button><button onclick="clearCurrent()">Làm lại ảnh</button></div>
<div><button onclick="exportJSON()">Tải annotations.json</button></div>
<div><label class="note">Nạp file annotation đã lưu: <input type="file" id="load" accept=".json,application/json"></label></div>
<p class="note">Mỗi ảnh phải được đánh dấu rõ: <b>Xác nhận nhãn áo</b>, <b>Ảnh không có áo</b>, hoặc <b>Không xác định được</b>. Không bấm “Ảnh không có áo” chỉ vì HSV rỗng. Sau khi gán nhãn, tải JSON và chuyển lại VM. Lưu JSON nhiều lần để tránh mất dữ liệu.</p>
<div id="msg" class="status"></div></div></div>
<script>
const META = __DATA__;
const IDSET = new Set(META.map(x=>x.id));
let index=0, drawPoints=[], image=new Image();
let ann={};
const canvas=document.getElementById('image'),ctx=canvas.getContext('2d');
function entry(){return ann[META[index].id]||(ann[META[index].id]={status:'pending',polygons:[]});}
function tell(s,bad=false){let e=document.getElementById('msg');e.textContent=s;e.className=bad?'danger':'status';}
function render(){const rec=META[index],a=entry();document.getElementById('header').textContent=`${index+1}/${META.length} · ${rec.id} · trạng thái: ${a.status} · đã duyệt: ${Object.values(ann).filter(x=>x.status!=='pending').length}`;
ctx.clearRect(0,0,256,256);if(image.complete&&image.naturalWidth)ctx.drawImage(image,0,0,256,256);
for(const p of a.polygons){if(!p.points.length)continue;ctx.beginPath();p.points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle=p.mode==='add'?'rgba(0,255,130,.22)':'rgba(255,130,0,.35)';ctx.fill();ctx.strokeStyle=p.mode==='add'?'#00e583':'#ff9f22';ctx.lineWidth=1.25;ctx.stroke();}
if(drawPoints.length){ctx.beginPath();drawPoints.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.strokeStyle='#ffeb3b';ctx.lineWidth=1.5;ctx.stroke();for(const [x,y] of drawPoints){ctx.beginPath();ctx.arc(x,y,2,0,Math.PI*2);ctx.fillStyle='#ffeb3b';ctx.fill();}}}
function loadImage(){drawPoints=[];image=new Image();image.onload=render;image.src=META[index].data;render();}
canvas.addEventListener('click',e=>{let r=canvas.getBoundingClientRect(),x=Math.max(0,Math.min(255,Math.round((e.clientX-r.left)*256/r.width))),y=Math.max(0,Math.min(255,Math.round((e.clientY-r.top)*256/r.height)));drawPoints.push([x,y]);render();});
function finish(){if(drawPoints.length<3){tell('Polygon cần ít nhất 3 điểm.',true);return;}entry().polygons.push({mode:document.getElementById('mode').value,points:drawPoints});drawPoints=[];entry().status='pending';render();}
function undoPoint(){drawPoints.pop();render();}function undoPolygon(){entry().polygons.pop();entry().status='pending';render();}
function mark(status){if(drawPoints.length){tell('Đóng polygon đang vẽ trước.',true);return;}let a=entry();if(status==='labeled'&&!a.polygons.some(p=>p.mode==='add')){tell('Cần vẽ ít nhất 1 polygon vùng áo.',true);return;}if(status==='empty'&&a.polygons.length){if(!confirm('Xóa polygon để đánh dấu ảnh KHÔNG có áo?'))return;a.polygons=[];}a.status=status;render();tell('Đã đánh dấu '+META[index].id);}
function clearCurrent(){if(confirm('Xóa nhãn hiện tại?')){ann[META[index].id]={status:'pending',polygons:[]};drawPoints=[];render();}}
function next(){index=Math.min(META.length-1,index+1);loadImage()}function prev(){index=Math.max(0,index-1);loadImage()}
function exportJSON(){const entries=META.map(m=>({sample_id:m.id,...(ann[m.id]||{status:'pending',polygons:[]})}));const payload={schema:'phyrc-garment-visible-mask-v1',annotations:entries};const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='garment_gt_annotations_v1.json';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);tell('Đã xuất '+entries.filter(r=>r.status!=='pending').length+'/'+META.length+' nhãn.');}
document.getElementById('load').addEventListener('change',async e=>{try{const f=e.target.files[0];if(!f)return;const j=JSON.parse(await f.text());if(j.schema!=='phyrc-garment-visible-mask-v1'||!Array.isArray(j.annotations))throw Error('Sai định dạng annotations');ann={};for(const a of j.annotations){if(IDSET.has(a.sample_id)){ann[a.sample_id]={status:a.status,polygons:a.polygons||[]};}}loadImage();tell('Đã nạp '+Object.keys(ann).length+' ảnh.');}catch(ex){tell(ex.message,true)}});
window.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;if(e.key==='ArrowRight')next();if(e.key==='ArrowLeft')prev();if(e.key==='Enter')finish();if(e.key==='Backspace')undoPoint();});
loadImage();
</script></body></html>'''


def make_html(benchmark, v3, output, count):
    rows = manifest_read(benchmark/'annotation_manifest.csv')
    data = json.loads((v3/'results.json').read_text(encoding='utf-8'))
    if not isinstance(data,list): raise ValueError('v3 results.json must be a list')
    picked = select_samples(rows,data,count)
    payload=[]
    for r in picked:
        path=(benchmark/r['rgb']).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        raw=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if raw is None or raw.shape[:2]!=(256,256): raise ValueError(f'Expected 256x256 RGB: {path}')
        encoded=base64.b64encode(path.read_bytes()).decode('ascii')
        payload.append({'id':r['sample_id'],'data':'data:image/png;base64,'+encoded})
    output.mkdir(parents=True,exist_ok=True)
    html=HTML.replace('__DATA__',json.dumps(payload,ensure_ascii=False))
    dest=output/'annotate_rgb_only.html'
    if dest.exists(): raise FileExistsError(f'Will not overwrite: {dest}')
    dest.write_text(html,encoding='utf-8')
    (output/'selected_samples.json').write_text(json.dumps({'schema':SCHEMA,'ids':[r['sample_id'] for r in picked]},indent=2),encoding='utf-8')
    print(f'Generated {dest} for {len(picked)} images; embedded RGB only; no GT created')
    print('Copy the standalone HTML to Windows; open locally in Chrome/Edge and export annotations JSON.')


def rasterize(annotation, sid, shape=(256,256)):
    status=annotation.get('status','pending')
    if status not in ('labeled','empty','unobservable','pending'):
        raise ValueError(f'{sid}: unknown status {status}')
    if status in ('unobservable','pending'):return None
    polygons=annotation.get('polygons',[])
    if not isinstance(polygons,list): raise ValueError(f'{sid}: invalid polygons')
    if status=='empty':
        if polygons: raise ValueError(f'{sid}: empty GT cannot have polygons')
        return np.zeros(shape,dtype=bool)
    if not any(p.get('mode')=='add' for p in polygons): raise ValueError(f'{sid}: labeled but has no add polygons')
    out=np.zeros(shape,np.uint8)
    # independent of draw order: union adds first, subtract holes second
    for mode,value in (('add',1),('erase',0)):
        for poly in polygons:
            if poly.get('mode') not in ('add','erase'): raise ValueError(f'{sid}: invalid polygon mode')
            if poly['mode']!=mode:continue
            pts=np.asarray(poly.get('points'),dtype=float)
            if pts.ndim!=2 or pts.shape[1]!=2 or len(pts)<3 or not np.isfinite(pts).all():
                raise ValueError(f'{sid}: polygon must have 3+ finite 2D points')
            if (pts<0).any() or (pts[:,0]>=shape[1]).any() or (pts[:,1]>=shape[0]).any():
                raise ValueError(f'{sid}: polygon point outside image bounds')
            cv2.fillPoly(out,[np.rint(pts).astype(np.int32)],value)
    if not out.any():raise ValueError(f'{sid}: labeled but resulting mask is empty')
    return out.astype(bool)


def load_mask(path, shape):
    if not path.is_file():return None
    im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if im is None or im.shape!=shape:raise ValueError(f'Invalid mask image: {path}')
    # all saved masks in benchmark use 0 / 255
    return im>127


def metrics(pred,gt):
    tp=int(np.count_nonzero(pred & gt));fp=int(np.count_nonzero(pred & ~gt));fn=int(np.count_nonzero(~pred & gt))
    union=tp+fp+fn
    if not gt.any():return {'gt_pixels':0,'pred_pixels':int(pred.sum()),'tp':tp,'fp':fp,'fn':fn,'iou':1.0 if not pred.any() else 0.0,'dice':1.0 if not pred.any() else 0.0,'precision':None,'recall':None}
    return {'gt_pixels':int(gt.sum()),'pred_pixels':int(pred.sum()),'tp':tp,'fp':fp,'fn':fn,
            'iou':tp/union if union else None,'dice':2*tp/(2*tp+fp+fn),
            'precision':tp/(tp+fp) if tp+fp else 0.0,'recall':tp/(tp+fn)}


def average(values):return round(float(np.mean(values)),5) if values else None


def evaluate(benchmark,v3,annot_path,output):
    selected=json.loads((output/'selected_samples.json').read_text(encoding='utf-8')) if (output/'selected_samples.json').is_file() else None
    manifest={r['sample_id']:r for r in manifest_read(benchmark/'annotation_manifest.csv')}
    records=json.loads((v3/'results.json').read_text(encoding='utf-8'))
    by_results={r['sample_id']:r for r in records}
    gt_data=json.loads(annot_path.read_text(encoding='utf-8'))
    if gt_data.get('schema')!=SCHEMA or not isinstance(gt_data.get('annotations'),list):raise ValueError('Invalid GT JSON schema')
    gt_list=gt_data['annotations']
    ids=[r.get('sample_id') for r in gt_list]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate annotation IDs')
    if set(ids)-set(manifest):raise ValueError('Unknown sample IDs in GT annotations')
    if selected and set(ids)!=set(selected['ids']):raise ValueError('GT IDs do not match selected_samples.json; wrong preparation set?')
    per=[];gt_masks={}; missing=[]
    methods=('hsv','sam_baseline','sam_v3_selected')
    for a in gt_list:
        sid=a['sample_id'];gt=rasterize(a,sid)
        if gt is None:continue
        rec=by_results.get(sid)
        if rec is None:raise ValueError(f'Missing v3 result for {sid}')
        source=manifest[sid]
        paths={'hsv':benchmark/source['hsv_prediction'],
               'sam_baseline':v3/'baseline_masks'/(sid+'.png'),
               'sam_v3_selected':v3/'selected_masks'/(sid+'.png')}
        pred_masks={key:load_mask(p,gt.shape) for key,p in paths.items()}
        if pred_masks['hsv'] is None:raise FileNotFoundError(paths['hsv'])
        if rec.get('status')=='ok' and (pred_masks['sam_baseline'] is None or pred_masks['sam_v3_selected'] is None):
            raise FileNotFoundError(f'{sid}: v3 marked ok, but SAM prediction file missing')
        if rec.get('status')!='ok' and (pred_masks['sam_baseline'] is not None or pred_masks['sam_v3_selected'] is not None):
            raise ValueError(f'{sid}: skipped sample unexpectedly has SAM masks')
        for method,mask in pred_masks.items():
            if mask is None:
                missing.append((sid,method));continue
            per.append({'sample_id':sid,'frame':int(source['frame']),'camera':source['camera'],
                        'gt_status':a['status'],'method':method,'v3_selection':rec.get('selected'),**metrics(mask,gt)})
        gt_masks[sid]=gt
    if not per:raise ValueError('No usable labeled samples: annotate and mark them first')
    output.mkdir(parents=True,exist_ok=True)
    gt_dir=output/'gt_masks_manual';gt_dir.mkdir(exist_ok=True)
    for sid,mask in gt_masks.items():
        png=gt_dir/(sid+'.png')
        if not cv2.imwrite(str(png),mask.astype(np.uint8)*255):raise OSError(png)
    fieldnames=['sample_id','frame','camera','gt_status','method','v3_selection','gt_pixels','pred_pixels','tp','fp','fn','iou','dice','precision','recall']
    with (output/'per_image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(per)
    summary={'purpose':'Human-labeled visible garment segmentation evaluation; same-episode internal test ONLY',
             'annotated_samples':len(gt_masks),'positive_gt_samples':sum(bool(m.any()) for m in gt_masks.values()),
             'empty_gt_samples':sum(not m.any() for m in gt_masks.values()),
             'unreviewed_or_unobservable':len(gt_list)-len(gt_masks),'missing_prediction_pairs':missing,'methods':{}}
    for method in methods:
        r=[x for x in per if x['method']==method]
        pos=[x for x in r if x['gt_pixels']>0]
        neg=[x for x in r if x['gt_pixels']==0]
        tp=sum(x['tp'] for x in pos);fp=sum(x['fp'] for x in pos);fn=sum(x['fn'] for x in pos)
        summary['methods'][method]={
            'evaluated_images':len(r),'positive_gt_images':len(pos),'empty_gt_images':len(neg),
            'macro_iou_positive_gt':average([x['iou'] for x in pos]),
            'macro_dice_positive_gt':average([x['dice'] for x in pos]),
            'macro_precision_positive_gt':average([x['precision'] for x in pos]),
            'macro_recall_positive_gt':average([x['recall'] for x in pos]),
            'micro_iou_positive_gt':round(tp/(tp+fp+fn),5) if pos and tp+fp+fn else None,
            'micro_precision_positive_gt':round(tp/(tp+fp),5) if pos and tp+fp else None,
            'micro_recall_positive_gt':round(tp/(tp+fn),5) if pos and tp+fn else None,
            'empty_gt_false_positive_images':sum(x['pred_pixels']>0 for x in neg),
            'empty_gt_total_false_positive_pixels':sum(x['fp'] for x in neg)}
    # Compare baseline-v3 on exact same positive GT images, never compare unpaired averages.
    pairs=defaultdict(dict)
    for r in per:
        if r['gt_pixels']>0:pairs[r['sample_id']][r['method']]=r['iou']
    paired=[d['sam_v3_selected']-d['sam_baseline'] for d in pairs.values() if 'sam_baseline' in d and 'sam_v3_selected' in d]
    summary['paired_v3_minus_baseline_iou']={'n':len(paired),'mean':average(paired),
        'improved':sum(x>1e-9 for x in paired),'worsened':sum(x<-1e-9 for x in paired),'unchanged':sum(abs(x)<=1e-9 for x in paired)}
    summary['limitations']=['Ground truth is human polygon approximation of garment VISIBLE pixels only.',
        'No model predictions were displayed during labeling; human labels still require quality review.',
        'SAM missing for HSV-empty samples is NA, NOT a zero-IoU prediction.',
        'Single demo_0: no independent episode generalization, no unseen test benchmark.',
        'Positive-GT macro IoU excludes empty GT; false positives on empty GT reported separately.']
    (output/'evaluation_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','evaluate'])
    parser.add_argument('--benchmark',required=True)
    parser.add_argument('--v3',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--count',type=int,default=25,help='Target number of images for prepare')
    parser.add_argument('--annotations',help='JSON exported by the browser, required for evaluate')
    a=parser.parse_args()
    benchmark=Path(a.benchmark).expanduser().resolve();v3=Path(a.v3).expanduser().resolve();out=Path(a.output).expanduser().resolve()
    if a.command=='prepare':
        if not 1<=a.count<=80:parser.error('--count must be between 1 and 80')
        make_html(benchmark,v3,out,a.count)
    else:
        if not a.annotations:parser.error('--annotations is required for evaluate')
        evaluate(benchmark,v3,Path(a.annotations).expanduser().resolve(),out)

if __name__=='__main__': main()
