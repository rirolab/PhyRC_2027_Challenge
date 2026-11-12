#!/usr/bin/env python3
"""PhyRC landmark heatmap prototype v3: SAM2.1 baseline RGB+mask supervised training.

Commands: prepare, train, evaluate, predict. Requires numpy, opencv-python, torch.
No SAM selected/v3-negative masks, ground-truth robot poses, or simulator state.
Only annotated positive visible keypoints are supervised; missing labels are UNKNOWN.
Data from one episode is NOT an independent validation of generalization.
"""
from __future__ import annotations
import argparse, csv, json, math, random, re
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

KEYS=['collar_center','left_sleeve_tip','right_sleeve_tip','hem_left','hem_right']
SWAP=[0,2,1,4,3]
SIDE=128
HEAT=64
SIGMA=1.75


def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def source_rows(bench):
    with (Path(bench)/'annotation_manifest.csv').open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def frame_group(row):
    sid=row['sample_id'];m=re.match(r'^(demo_\d+)_\d+_',sid)
    return f"{m.group(1) if m else sid}:{row['frame']}"


def check_gt(label):
    if not isinstance(label.get('points'),dict):raise ValueError('points should be a dict')
    for k,xy in label['points'].items():
        if k not in KEYS:raise ValueError('Unknown landmark '+k)
        if not isinstance(xy,(list,tuple)) or len(xy)!=2 or not all(isinstance(t,(int,float)) and math.isfinite(t) and 0<=t<=255 for t in xy):
            raise ValueError(f'Invalid coordinate {k}={xy}')


def prepare(a):
    bench=Path(a.benchmark).resolve();sam=Path(a.sam).resolve()
    labels=load(a.labels)
    if labels.get('schema')!='phyrc-landmarks-visible-v1':raise ValueError('Unexpected label schema')
    by_id={}
    for l in labels['labels']:
        sid=l['sample_id']
        if sid in by_id:raise ValueError('Duplicate '+sid)
        check_gt(l);by_id[sid]=l
    entries=[];count=Counter();missed=[]
    for row in source_rows(bench):
        sid=row['sample_id'];label=by_id.get(sid)
        mp=sam/'baseline_masks'/f'{sid}.png';rp=bench/row['rgb']
        if not mp.is_file() or not rp.is_file():
            if label and label.get('points'):missed.append(sid)
            continue
        rgb=cv2.imread(str(rp),cv2.IMREAD_COLOR);mask=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None or rgb.shape[:2]!=(256,256) or mask.shape!=(256,256):raise ValueError('Invalid image '+sid)
        is_reviewed=bool(label and label.get('reviewed'))
        points=label.get('points',{}) if is_reviewed else {}
        if points:
            for k in points:count[k]+=1
        entries.append({'sample_id':sid,'frame':int(row['frame']),'camera':row['camera'],
                        'group':frame_group(row),'rgb':row['rgb'],
                        'mask':f'baseline_masks/{sid}.png','reviewed':is_reviewed,
                        'points':points,'observability':label.get('observability','unknown') if is_reviewed else 'unknown'})
    positive=[x for x in entries if x['points']]
    groups=sorted({x['group'] for x in positive})
    if not positive:raise ValueError('No positive labeled samples with SAM masks')
    # Fixed deterministic group split, all cameras at same frame stay in same set.
    rng=random.Random(a.seed); rng.shuffle(groups)
    nval=max(1,int(round(len(groups)*a.val_fraction))) if len(groups)>=2 and a.val_fraction>0 else 0
    nval=min(nval,len(groups)-1)
    vg=set(groups[:nval]);tg=set(groups[nval:])
    for x in entries:
        x['split']='val' if x['group'] in vg else ('train' if x['group'] in tg else 'unlabeled')
    # Only reviewed nonempty labels contribute supervised heatmap loss.
    assert not (vg & tg)
    index={'schema':'phyrc-keypoint-supervised-v3','benchmark_root':str(bench),'sam_root':str(sam),
           'keys':KEYS,'seed':a.seed,'val_fraction':a.val_fraction,'grouping':'demo+frame',
           'entries':entries,'training_limitations':['Single source episode: split by frame is correlated; not independent episode generalization.',
            'Missing keypoints are unknown/unlabeled, NOT negative examples.',
            'SAM segmentation errors are inherited; wearer left/right labels require independent review.']}
    dump(a.output,index)
    print(json.dumps({'index':a.output,'sam_image_count':len(entries),'labeled_images':len(positive),
        'landmarks':dict(count),'labeled_missing_sam':missed,
        'train_images':sum(bool(x['points']) and x['split']=='train' for x in entries),
        'val_images':sum(bool(x['points']) and x['split']=='val' for x in entries),
        'train_frame_groups':len(tg),'val_frame_groups':len(vg)},indent=2))


class KPData(Dataset):
    def __init__(self,index,split=None,augment=False):
        self.index=index;self.entries=[x for x in index['entries'] if (split is None or x['split']==split) and (split is None or x['points'])]
        self.augment=augment;self.bench=Path(index['benchmark_root']);self.sam=Path(index['sam_root'])
        if split in ('train','val') and not self.entries:raise ValueError('No labeled images in '+split)
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        r=self.entries[i];rgb=cv2.imread(str(self.bench/r['rgb']),cv2.IMREAD_COLOR)
        mask=cv2.imread(str(self.sam/r['mask']),cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:raise FileNotFoundError(r['sample_id'])
        rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(rgb,(SIDE,SIDE),interpolation=cv2.INTER_AREA)
        mask=cv2.resize(mask,(SIDE,SIDE),interpolation=cv2.INTER_NEAREST)>127
        coords=np.zeros((5,2),dtype=np.float32);valid=np.zeros(5,dtype=np.float32)
        for j,k in enumerate(KEYS):
            if k in r['points']:
                coords[j]=np.asarray(r['points'][k],dtype=np.float32)*(SIDE/256)
                valid[j]=1
        if self.augment:
            if random.random()<.5:
                rgb=rgb[:,::-1].copy();mask=mask[:,::-1].copy()
                coords[:,0]=(SIDE-1)-coords[:,0]
                coords=coords[SWAP].copy();valid=valid[SWAP].copy()
            if random.random()<.75:
                scale=random.uniform(.85,1.15)
                rgb=np.clip(rgb.astype(np.float32)*scale,0,255).astype('uint8')
        # Preserve context, but downweight outside SAM to favor garment pixels.
        x=np.concatenate([rgb.astype(np.float32)/255*(mask[:,:,None]*.85+.15),mask.astype(np.float32)[:,:,None]],axis=2)
        x=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1).astype(np.float32)))
        # Heatmap distribution is normalized; masked KL/cross entropy ignores every missing key.
        gy,gx=np.mgrid[0:HEAT,0:HEAT]
        heat=np.zeros((5,HEAT,HEAT),dtype=np.float32)
        for j in np.flatnonzero(valid):
            cx=coords[j,0]*(HEAT/SIDE);cy=coords[j,1]*(HEAT/SIDE)
            h=np.exp(-((gx-cx)**2+(gy-cy)**2)/(2*SIGMA**2))
            heat[j]=h/(np.sum(h)+1e-12)
        return x,torch.from_numpy(heat),torch.from_numpy(valid),r['sample_id']


class Block(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.GroupNorm(4,co),nn.ReLU(inplace=True),
                                  nn.Conv2d(co,co,3,padding=1,bias=False),nn.GroupNorm(4,co),nn.ReLU(inplace=True))
    def forward(self,x):return self.layers(x)


class KeypointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=Block(4,16);self.enc2=Block(16,32);self.enc3=Block(32,64);self.mid=Block(64,96)
        self.up2=Block(96+64,64);self.up1=Block(64+32,32)
        self.out=nn.Conv2d(32,5,1)
    def forward(self,x):
        a=self.enc1(x);b=self.enc2(nn.functional.avg_pool2d(a,2));c=self.enc3(nn.functional.avg_pool2d(b,2))
        z=self.mid(nn.functional.avg_pool2d(c,2));z=nn.functional.interpolate(z,size=c.shape[-2:],mode='bilinear',align_corners=False)
        z=self.up2(torch.cat([z,c],dim=1));z=nn.functional.interpolate(z,size=b.shape[-2:],mode='bilinear',align_corners=False)
        z=self.up1(torch.cat([z,b],dim=1));return self.out(z)


def loss_fn(logits,targets,visible):
    lp=torch.log_softmax(logits.flatten(2),dim=-1)
    ce=-(targets.flatten(2)*lp).sum(dim=-1)
    return (ce*visible).sum()/visible.sum().clamp_min(1)


def decode(logits,trained):
    # Pixel heatmap is 64x64. max softmax probability is NOT a calibrated confidence.
    b,k,h,w=logits.shape
    probs=torch.softmax(logits.flatten(2),dim=-1)
    peak,idx=probs.max(dim=-1)
    out=[]
    for n in range(b):
        pts={};scores={}
        for j,key in enumerate(KEYS):
            if trained[j]==0:continue
            yy=int(idx[n,j]//w);xx=int(idx[n,j]%w)
            pts[key]=[float((xx+.5)*(256/w)),float((yy+.5)*(256/h))]
            scores[key]=round(float(peak[n,j]),6)
        out.append((pts,scores))
    return out


def device_for(name):
    if name=='auto':return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def run_epoch(model,loader,dev,opt=None):
    model.train(opt is not None);loss_total=0.;seen=0
    ctx=torch.enable_grad() if opt is not None else torch.no_grad()
    with ctx:
        for x,t,v,_ in loader:
            x=x.to(dev);t=t.to(dev);v=v.to(dev)
            out=model(x);l=loss_fn(out,t,v)
            if opt is not None:
                opt.zero_grad(set_to_none=True);l.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            loss_total+=float(l.detach())*len(x);seen+=len(x)
    return round(loss_total/max(seen,1),5)


def train(a):
    idx=load(a.data);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    dev=device_for(a.device)
    tr=KPData(idx,'train',True);dl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=0)
    va=[r for r in idx['entries'] if r['split']=='val' and r['points']]
    val=DataLoader(KPData(idx,'val',False),batch_size=a.batch_size,shuffle=False,num_workers=0) if va else None
    counts=Counter(k for e in tr.entries for k in e['points'])
    model=KeypointNet().to(dev)
    optim=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    best=float('inf');history=[]
    for ep in range(1,a.epochs+1):
        tl=run_epoch(model,dl,dev,optim);vl=run_epoch(model,val,dev) if val else None
        history.append({'epoch':ep,'train_loss':tl,'val_loss':vl})
        score=vl if vl is not None else tl
        if score<best:
            best=score
            ckpt={'state_dict':model.state_dict(),'keys':KEYS,'image_size':SIDE,'heatmap_size':HEAT,
                  'train_key_counts':dict(counts),'epoch':ep,'seed':a.seed,
                  'source_episode_warning':'single episode; not a generalization guarantee'}
            torch.save(ckpt,out/'best.pt')
        if ep==1 or ep%max(1,a.epochs//10)==0 or ep==a.epochs:
            print(f'epoch {ep:3d}/{a.epochs} train={tl:.5f} val={vl}')
    dump(out/'train_history.json',history)
    dump(out/'training_summary.json',{'train_images':len(tr),'val_images':len(va),'train_key_counts':dict(counts),
      'epochs':a.epochs,'best_selection_loss':best,'device':str(dev),'checkpoint':str(out/'best.pt'),
      'warning':'validation samples are correlated: same demo; no independent episode testing'})
    print('Saved:',out/'best.pt')


def get_model(path,dev):
    # state_dict-only payload generated by this script; use only trusted checkpoints.
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('keys')!=KEYS:raise ValueError('Checkpoint key schema mismatch')
    model=KeypointNet().to(dev);model.load_state_dict(ck['state_dict']);model.eval()
    counts=[int(ck['train_key_counts'].get(k,0)) for k in KEYS]
    return model,counts,ck


def metrics(records,annotations):
    by_id={x['sample_id']:x for x in records};errors=[];per={k:{'gt':0,'pred':0,'hit5':0,'hit10':0,'error':[]} for k in KEYS}
    count=0;scored=0
    for r in annotations:
        count+=1
        p=by_id.get(r['sample_id'],{}).get('points_2d',{})
        for k,xy in r['points'].items():
            x=per[k];x['gt']+=1;scored+=1
            if k not in p:continue
            d=float(np.linalg.norm(np.asarray(p[k])-np.asarray(xy)))
            x['pred']+=1;x['hit5']+=d<=5;x['hit10']+=d<=10;x['error'].append(d);errors.append(d)
    cov=sum(v['pred'] for v in per.values());p5=sum(v['hit5'] for v in per.values());p10=sum(v['hit10'] for v in per.values())
    return {'labeled_images':count,'gt_landmarks':scored,'predicted_at_gt':cov,
      'coverage':round(cov/max(scored,1),5),'pck5_over_all':round(p5/max(scored,1),5),'pck10_over_all':round(p10/max(scored,1),5),
      'pck5_conditional':round(p5/cov,5) if cov else None,'pck10_conditional':round(p10/cov,5) if cov else None,
      'mean_error_px':round(float(np.mean(errors)),3) if errors else None,
      'per_key':{k:{t:v for t,v in x.items() if t!='error'}|{'mean_error_px':round(float(np.mean(x['error'])),3) if x['error'] else None} for k,x in per.items()},
      'limitations':['Missing labels are unknown, not ground-truth absent landmarks.',
                     'Same demo_0 as training; validation by timestamp is correlated; no independent episode generalization.',
                     'Wearer-left/right correctness depends on annotation convention and camera view.']}


def infer(a,split=None):
    index=load(a.data);dev=device_for(a.device);model,trained,ck=get_model(a.checkpoint,dev)
    data=KPData(index,split=None,augment=False)
    if split:
        data.entries=[x for x in data.entries if x['split']==split and x['points']]
        if not data.entries:raise ValueError('No labeled samples in '+split)
    dl=DataLoader(data,batch_size=a.batch_size,shuffle=False,num_workers=0)
    preds=[];lookup={x['sample_id']:x for x in data.entries}
    with torch.inference_mode():
        for img,_,_,sids in dl:
            logits=model(img.to(dev))
            for sid,(pts,scores) in zip(sids,decode(logits,trained)):
                # Conservative observed-mask gate: do not emit a landmark on background.
                mask_path=Path(index['sam_root'])/lookup[sid]['mask']
                mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)>127
                rejected=[]
                for key,xy in list(pts.items()):
                    cx,cy=[int(round(z)) for z in xy]
                    x0,x1=max(0,cx-2),min(256,cx+3)
                    y0,y1=max(0,cy-2),min(256,cy+3)
                    if not (0<=cx<256 and 0<=cy<256 and bool(mask[y0:y1,x0:x1].any())):
                        rejected.append(key);del pts[key];scores.pop(key,None)
                preds.append({'sample_id':sid,'points_2d':pts,'peak_probability_uncalibrated':scores,
                              'rejected_outside_sam_mask':rejected,
                              'status':'heatmap_proposal' if pts else 'no_accepted_landmarks',
                              'front_back':'unknown','landmark_semantics':'model_hypotheses',
                              'camera':lookup[sid]['camera'],'frame':lookup[sid]['frame']})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    dump(out/'landmark_predictions_v3.json',preds)
    if split:
        summary=metrics(preds,data.entries)
        summary.update({'split':split,'checkpoint_epoch':ck['epoch'],'images_evaluated':len(data.entries)})
        dump(out/'evaluation_v3.json',summary)
        print(json.dumps(summary,indent=2))
    else:
        dump(out/'prediction_summary_v3.json',{'total':len(preds),'train_key_counts':dict(zip(KEYS,trained)),
          'front_back_classification':'not implemented','not_controller_ready':True})
        print('Predictions:',len(preds),'in',out)
    if a.previews:
        preview=out/'previews';preview.mkdir(exist_ok=True)
        for r in preds:
            meta=lookup[r['sample_id']];src=Path(index['benchmark_root'])/meta['rgb']
            image=cv2.imread(str(src));
            if image is None:continue
            for j,k in enumerate(KEYS):
                if k not in r['points_2d']:continue
                x,y=map(lambda t:int(round(t)),r['points_2d'][k]);c=[(0,80,255),(0,255,0),(0,200,255),(255,90,90),(255,0,255)][j]
                cv2.circle(image,(x,y),4,c,-1)
                cv2.putText(image,k[:6],(min(x+4,205),max(y-5,12)),cv2.FONT_HERSHEY_SIMPLEX,.32,c,1)
            cv2.imwrite(str(preview/(r['sample_id']+'.png')),image)


def main():
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('prepare',help='Validate and index labeled dataset, grouped by frame')
    for k in ['benchmark','sam','labels','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--val-fraction',type=float,default=.25);p.set_defaults(fn=prepare)
    p=sp.add_parser('train',help='Train small heatmap network with visible-point masking')
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto');p.set_defaults(fn=train)
    for cmd in ('evaluate','predict'):
        p=sp.add_parser(cmd,help='Evaluate on labeled split or predict all SAM images')
        p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True)
        p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=8)
        p.add_argument('--device',default='auto');p.add_argument('--previews',action='store_true')
        if cmd == 'evaluate':
            p.add_argument('--split', choices=['train', 'val'], default='val')
        p.set_defaults(fn=infer, split=None if cmd=='predict' else 'val')
    a=ap.parse_args()
    if a.cmd=='prepare' and not (0<=a.val_fraction<1):ap.error('val-fraction must be [0,1)')
    if a.cmd=='train' and (a.epochs<1 or a.batch_size<1 or a.lr<=0):ap.error('invalid train hyperparameters')
    a.fn(a) if a.cmd in ('prepare','train') else a.fn(a,split=a.split)
if __name__=='__main__':main()
