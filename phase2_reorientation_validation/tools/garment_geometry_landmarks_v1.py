#!/usr/bin/env python3
"""PhyRC garment geometry and landmark research v1 (public RGB-D only).

prepare: build offline HTML for independent human semantic landmark annotation.
geometry: compute optical-frame sparse RGB-D geometry from SAM 2.1 baseline,
          and optionally 3D coordinates for human-visible landmarks.
Does NOT automatically classify front/back or name silhouette extrema as sleeves.
"""
from __future__ import annotations
import argparse, base64, csv, json, math
from pathlib import Path
import numpy as np
import cv2

KEYS=['collar_center','left_sleeve_tip','right_sleeve_tip','hem_left','hem_right']
CAMS=['overview','robot_0_wrist','robot_1_wrist','robot_0_head','robot_1_head']
HFOV={'overview':60.,'robot_0_wrist':70.,'robot_1_wrist':70.,'robot_0_head':70.,'robot_1_head':70.}

def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def putjson(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def mask_path(sam,sid): return sam/'baseline_masks'/f'{sid}.png'

def getmask(sam,sid):
    p=mask_path(sam,sid)
    if not p.exists(): return None
    m=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
    if m is None or m.shape!=(256,256): raise ValueError(f'Invalid SAM mask: {p}')
    return m>127

HTML=r'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>PhyRC Landmark Labels v1</title>
<style>body{background:#101827;color:#f5f7ff;font:16px Arial;margin:18px}.layout{display:flex;gap:30px;flex-wrap:wrap}canvas{width:640px;height:640px;max-width:90vw;image-rendering:pixelated;cursor:crosshair}button,select{background:#233149;color:white;border:1px solid #7183a2;border-radius:6px;padding:10px;margin:4px;cursor:pointer}.text{max-width:450px;line-height:1.6}small{color:#adbeda}.warn{color:#ffc97a}</style>
<h2>PhyRC · Semantic Garment Landmarks v1</h2><p>Gán nhãn trên RGB độc lập. Điểm không nhìn thấy: để trống, KHÔNG đoán. Left/right = bên trái/phải của ÁO khi nhìn từ mặt trước; nếu không phân biệt được, để trống. Front/back chỉ chọn nếu có bằng chứng thị giác.</p><div id="status"></div><div class="layout"><div><canvas id="c" width="256" height="256"></canvas><p><small>Toạ độ nguồn 256×256, màn hình được phóng lớn.</small></p></div><div class="text"><div><button onclick="move(-1)">← Trước</button><button onclick="move(1)">Tiếp →</button></div><div><select id="key" onchange="paint()"><option value="collar_center">Cổ áo (giữa, nếu thấy)</option><option value="left_sleeve_tip">Đầu tay áo TRÁI của áo</option><option value="right_sleeve_tip">Đầu tay áo PHẢI của áo</option><option value="hem_left">Gấu trái của áo</option><option value="hem_right">Gấu phải của áo</option></select><button onclick="clearPoint()">Xóa điểm chọn</button></div><div><label>Mặt áo: <select id="face" onchange="setFace(this.value)"><option value="unknown">unknown</option><option value="front">front</option><option value="back">back</option></select></label></div><div><label>Quan sát: <select id="visible" onchange="setVisibility(this.value)"><option value="partial">partial/occluded</option><option value="sufficient">sufficient</option><option value="none">no garment visible</option></select></label></div><div><button onclick="save()">Xác nhận ảnh</button><button onclick="exportJ()">Tải landmark_labels_v1.json</button></div><div><label>Nạp file đã lưu: <input id="load" type="file" accept=".json" /></label></div><p id="msg" class="warn"></p><p><small>Nhấn lần lượt lên ảnh để đặt keypoint đã chọn. Bấm nút xác nhận cho mỗi ảnh, kể cả ảnh không thể xác định landmark. Đây là HUMAN LABELS dùng để phát triển và đánh giá thuật toán, chưa phải kết quả tự động.</small></p></div></div>
<script>
const data=__EMBED__;const K=['collar_center','left_sleeve_tip','right_sleeve_tip','hem_left','hem_right'];let idx=0;let labels={},image=new Image();const c=document.getElementById('c'),ctx=c.getContext('2d');const palette=['#ff4545','#3dff71','#ffe344','#56b5ff','#ff70eb'];
function cur(){const id=data[idx].id;return labels[id]||(labels[id]={points:{},front_back:'unknown',observability:'partial',reviewed:false});}
function paint(){const item=data[idx],l=cur();document.getElementById('status').textContent=`${idx+1}/${data.length} · ${item.id} · đã duyệt ${Object.values(labels).filter(x=>x.reviewed).length}`;document.getElementById('face').value=l.front_back;document.getElementById('visible').value=l.observability;ctx.clearRect(0,0,256,256);if(image.complete&&image.naturalWidth)ctx.drawImage(image,0,0,256,256);for(let i=0;i<K.length;i++){let p=l.points[K[i]];if(!p)continue;ctx.beginPath();ctx.arc(p[0],p[1],3,0,Math.PI*2);ctx.strokeStyle='#000';ctx.lineWidth=3;ctx.stroke();ctx.fillStyle=palette[i];ctx.fill();ctx.fillStyle='#111';ctx.fillRect(Math.min(217,p[0]+5),Math.max(1,p[1]-9),32,12);ctx.fillStyle=palette[i];ctx.font='10px Arial';ctx.fillText(String(i+1),p[0]+8,p[1]);}}
function load(){image=new Image();image.onload=paint;image.src=data[idx].rgb;paint()};function move(d){idx=Math.max(0,Math.min(data.length-1,idx+d));load()};
c.addEventListener('click',e=>{let r=c.getBoundingClientRect();let x=Math.max(0,Math.min(255,Math.round((e.clientX-r.left)*256/r.width)));let y=Math.max(0,Math.min(255,Math.round((e.clientY-r.top)*256/r.height)));cur().points[document.getElementById('key').value]=[x,y];cur().reviewed=false;paint()});
function clearPoint(){delete cur().points[document.getElementById('key').value];cur().reviewed=false;paint()};function setFace(v){cur().front_back=v;cur().reviewed=false;paint()};function setVisibility(v){cur().observability=v;cur().reviewed=false;paint()};
function save(){cur().reviewed=true;paint();document.getElementById('msg').textContent='Đã xác nhận '+data[idx].id;};function exportJ(){const obj={schema:'phyrc-landmarks-visible-v1',labels:data.map(x=>({sample_id:x.id,...(labels[x.id]||{points:{},front_back:'unknown',observability:'partial',reviewed:false})}))};const b=new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='landmark_labels_v1.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),2000)};
document.getElementById('load').onchange=async e=>{try{const obj=JSON.parse(await e.target.files[0].text());if(obj.schema!=='phyrc-landmarks-visible-v1')throw Error('Wrong schema');labels={};for(const l of obj.labels||[])if(data.some(x=>x.id===l.sample_id))labels[l.sample_id]={points:l.points||{},front_back:l.front_back||'unknown',observability:l.observability||'partial',reviewed:!!l.reviewed};load()}catch(err){document.getElementById('msg').textContent=err.message}};load();
</script></html>'''

def prepare(args):
    benchmark=Path(args.benchmark);sam=Path(args.sam);out=Path(args.output)
    manifest=rows(benchmark/'annotation_manifest.csv')
    ids=None
    if args.selection:
        d=json.loads(Path(args.selection).read_text(encoding='utf-8'));ids=set(d['ids'])
    payload=[]
    for r in manifest:
        sid=r['sample_id'];p=mask_path(sam,sid)
        if ids is not None and sid not in ids:continue
        if not p.exists():continue
        ip=benchmark/r['rgb'];raw=ip.read_bytes()
        payload.append({'id':sid,'rgb':'data:image/png;base64,'+base64.b64encode(raw).decode('ascii')})
    if not payload:raise ValueError('No matching RGB+SAM pairs')
    out.mkdir(parents=True,exist_ok=True)
    p=out/'annotate_landmarks_rgb_only.html'
    if p.exists() and not args.overwrite:raise FileExistsError(f'File exists: {p} (use --overwrite to regenerate)')
    p.write_text(HTML.replace('__EMBED__',json.dumps(payload)),encoding='utf-8')
    putjson(out/'prepared_samples.json',{'samples':[x['id'] for x in payload],'count':len(payload)})
    print(f'Annotation HTML: {p}; images={len(payload)}; RGB ONLY; no predictions shown')


def geom(mask,depth,valid,camera):
    ok=mask&valid&np.isfinite(depth)&(depth>0)
    yy,xx=np.nonzero(ok)
    if len(xx)<20:return {'status':'unavailable','reason':'insufficient_valid_mask_depth','valid_points':len(xx)}
    z=depth[yy,xx];h,w=mask.shape;fx=w/(2*math.tan(math.radians(HFOV[camera])/2));fy=fx;cx=w/2;cy=h/2
    pts=np.column_stack(((xx-cx)*z/fx,(yy-cy)*z/fy,z))
    # deterministic, bounded number of points; no world pose inferred
    if len(pts)>2048:pts=pts[np.linspace(0,len(pts)-1,2048,dtype=int)]
    center=pts.mean(axis=0);cov=np.cov(pts.T);evals,evecs=np.linalg.eigh(cov);sort=np.argsort(evals)[::-1]
    return {'status':'ok','frame':'camera_optical','valid_points':int(len(xx)),
       'centroid_xyz_m':center.round(6).tolist(),'covariance_m2':cov.round(9).tolist(),
       'pca_eigenvalues_m2':evals[sort].round(9).tolist(),
       'principal_axis_optical':evecs[:,sort[0]].round(6).tolist(),
       'depth_valid_fraction_inside_mask':round(float(len(xx)/max(int(mask.sum()),1)),5)}

def backproject_at(xy,mask,depth,valid,camera,radius):
    x,y=map(int,xy);h,w=mask.shape
    if not(0<=x<w and 0<=y<h):return {'status':'invalid_pixel'}
    if not mask[y,x]:return {'status':'outside_sam_mask','xy':[x,y]}
    yy,xx=np.mgrid[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]
    keep=valid[yy,xx]&mask[yy,xx]&np.isfinite(depth[yy,xx])&(depth[yy,xx]>0)
    if not keep.any():return {'status':'missing_local_depth','xy':[x,y]}
    # use median Z within nearby garment pixels; explicit approximation
    z=float(np.median(depth[yy[keep],xx[keep]]));fx=w/(2*math.tan(math.radians(HFOV[camera])/2));fy=fx
    return {'status':'ok','xy':[x,y],'xyz_optical_m':[round((x-w/2)*z/fx,6),round((y-h/2)*z/fy,6),round(z,6)],'median_depth_window_radius_px':radius}

def run(args):
    bench=Path(args.benchmark);sam=Path(args.sam);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=rows(bench/'annotation_manifest.csv');index={r['sample_id']:r for r in manifest}
    labels={}
    if args.labels:
        d=json.loads(Path(args.labels).read_text(encoding='utf-8'))
        if d.get('schema')!='phyrc-landmarks-visible-v1':raise ValueError('Invalid landmark JSON schema')
        for a in d['labels']:
            sid=a['sample_id'];
            if sid in labels:raise ValueError('Duplicate label '+sid)
            if sid not in index:raise ValueError('Label not in manifest: '+sid)
            labels[sid]=a
    results=[]
    for r in manifest:
        sid=r['sample_id'];mask=getmask(sam,sid)
        if mask is None:
            results.append({'sample_id':sid,'status':'no_sam_baseline','front_back':'unknown','landmarks':{}});continue
        dp=bench/r['depth_valid'] if r['depth_valid'] else None
        if dp is None or not dp.exists():
            results.append({'sample_id':sid,'status':'missing_depth','front_back':'unknown','landmarks':{}});continue
        with np.load(dp) as z:
            depth=np.asarray(z['depth_m'],dtype=np.float32);valid=np.asarray(z['depth_valid'],dtype=bool)
        if depth.shape!=(256,256) or valid.shape!=depth.shape:raise ValueError('Depth dimensions mismatch '+sid)
        g=geom(mask,depth,valid,r['camera']);label=labels.get(sid,{})
        label_ok=bool(label.get('reviewed'))
        fb=label.get('front_back','unknown') if label_ok else 'unknown'
        if fb not in ('front','back','unknown'):raise ValueError('Invalid front/back label '+sid)
        points={}
        if label_ok:
            for key,p in label.get('points',{}).items():
                if key not in KEYS:raise ValueError('Unknown landmark '+key)
                if not isinstance(p,list) or len(p)!=2:raise ValueError('Invalid xy '+sid)
                points[key]=backproject_at(p,mask,depth,valid,r['camera'],args.radius)
        results.append({'sample_id':sid,'status':'ok','camera':r['camera'],'frame':int(r['frame']),
          'sam_mask_pixels':int(mask.sum()),'geometry':g,
          'front_back':fb,'front_back_source':'human_visual_label' if label_ok and fb!='unknown' else 'unknown',
          'observability':label.get('observability','unknown') if label_ok else 'unknown',
          'landmarks':points,'landmark_source':'human_visible_points' if label_ok else 'none'})
    putjson(out/'garment_geometry_landmarks_v1.json',results)
    summary={'samples':len(results),'geometry_ok':sum(x.get('geometry',{}).get('status')=='ok' for x in results),
      'with_reviewed_landmarks':sum(x.get('landmark_source')=='human_visible_points' for x in results),
      'front_labeled':sum(x['front_back']=='front' for x in results),'back_labeled':sum(x['front_back']=='back' for x in results),
      'unknown_orientation':sum(x['front_back']=='unknown' for x in results),
      'notes':['SAM baseline only; no learned semantic landmark or orientation predictor.',
               '3D is CAMERA OPTICAL frame, NOT world frame. Intrinsics assume stated HFOV; verify against official camera parameters.',
               'Geometry includes occluding mannequin false positives where segmentation is wrong.',
               'Human semantic labels are training/evaluation labels, not automatic inference.']}
    putjson(out/'geometry_summary.json',summary);print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('prepare');a.add_argument('--benchmark',required=True);a.add_argument('--sam',required=True);a.add_argument('--output',required=True);a.add_argument('--selection',help='segmentation_eval_v1/selected_samples.json to reuse same subset');a.add_argument('--overwrite',action='store_true');a.set_defaults(func=prepare)
    a=s.add_parser('geometry');a.add_argument('--benchmark',required=True);a.add_argument('--sam',required=True);a.add_argument('--output',required=True);a.add_argument('--labels');a.add_argument('--radius',type=int,default=2);a.set_defaults(func=run)
    args=p.parse_args();
    if args.cmd=='geometry' and not(0<=args.radius<=10):p.error('--radius must be 0..10')
    args.func(args)
if __name__=='__main__':main()
