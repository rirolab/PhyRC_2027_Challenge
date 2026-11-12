"""Low-rate, synchronized 5-RGB-camera video evidence emitted by the policy.

One JPEG montage per sampled *policy observation*, plus index CSV. This does
not read audit/simulator ground truth. Encode MP4 AFTER evaluation with ffmpeg.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERAS = ("OVERVIEW", "R0 WRIST", "R1 WRIST", "R0 HEAD", "R1 HEAD")

class VideoCapture:
    def __init__(self, run_dir, *, stride=4, quality=78):
        self.root=Path(run_dir)/'video_frames'
        self.root.mkdir(parents=True,exist_ok=True)
        self.stride=max(1,int(stride));self.quality=max(30,min(int(quality),95))
        self.count=0;self.errors=0
        self.index=self.root/'frame_index.csv'
        with self.index.open('w',newline='',encoding='utf-8') as f:
            csv.writer(f).writerow(['video_frame','policy_frame','simulation_time_s','v1_mode',
                                     'handoff','scene_side','yielding_robot','passing_robot',
                                     'strategy_status','planner_status','candidate_family'])

    def push(self,obs,frame,*,mode,holding,intent=None,decision=None):
        if frame%self.stride:return
        try:
            rgb=np.asarray(obs['rgb'])
            if rgb.shape!=(5,256,256,3):
                raise ValueError('RGB shape mismatch '+str(rgb.shape))
            t=float(np.asarray(obs['simulation_time_s']).reshape(-1)[0])
            intent=intent or {};decision=decision or {}
            canvas=Image.new('RGB',(1344,586),(26,30,36))
            d=ImageDraw.Draw(canvas)
            headline=f"PhyRC V2 PLAN ONLY | policy frame={frame} | sim={t:.2f}s | V1={mode} | handoff={int(holding)}"
            d.text((12,8),headline,fill=(242,245,248))
            side=intent.get('garment_side','UNKNOWN');y=intent.get('yielding_robot');p=intent.get('passing_robot')
            role=f"scene={side} / {intent.get('scene_evidence','UNKNOWN')} | role=R{y} yields -> R{p} passes" if y is not None else f"scene={side} | ROLE UNKNOWN (not inferred)"
            d.text((12,31),role,fill=(241,195,94) if y is not None else (236,126,126))
            d.text((12,53),f"SDL intent={intent.get('status','WAITING')} | planner={decision.get('status','WAITING')} | proposed={bool(decision.get('best'))} | {decision.get('role_plan_status','NO_ROLE_PLAN_YET')} | NO V2 BASE ACTION",fill=(208,215,218))
            # Left overview 512 x 512; right 2x2 montage 256x256 each.
            for i in range(5):
                im=Image.fromarray(rgb[i].astype(np.uint8),mode='RGB')
                if i==0:
                    canvas.paste(im.resize((512,512)),(0,74))
                    d.text((9,82),'OVERVIEW (VIDEO AUDIT ONLY)',fill='white',stroke_width=1,stroke_fill='black')
                else:
                    j=i-1;x=512+256*(j%2);y0=74+256*(j//2)
                    canvas.paste(im,(x,y0))
                    d.text((x+6,y0+7),CAMERAS[i],fill='white',stroke_width=1,stroke_fill='black')
            # Independent XY-only plan sketch for operator diagnosis, NOT a
            # costmap or certified space-time robot trajectory.
            left=1034;right=1334;top=100;bottom=440
            d.rectangle((left,top,right,bottom),outline=(100,114,127),width=2)
            d.text((left+4,78),'COUPLED BASE PLAN (XY PROXY)',fill=(208,225,240))
            best=decision.get('best') if isinstance(decision,dict) else None
            wp=np.asarray((best or {}).get('waypoints',[]),float)
            if wp.ndim==3 and wp.shape[1:]==(2,2) and len(wp)>0 and np.isfinite(wp).all():
                lo=wp.reshape(-1,2).min(axis=0)-.25
                hi=wp.reshape(-1,2).max(axis=0)+.25
                size=np.maximum(hi-lo,.5)
                def place(v):
                    x=float((v[0]-lo[0])/size[0]);y=float((v[1]-lo[1])/size[1])
                    return (round(left+10+x*(right-left-20)),round(bottom-10-y*(bottom-top-20)))
                for r,color in ((0,(73,174,244)),(1,(249,163,74))):
                    xy=[place(v[r]) for v in wp]
                    if len(xy)>1:d.line(xy,fill=color,width=3,joint='curve')
                    for point in (xy[0],xy[-1]):
                        d.ellipse((point[0]-4,point[1]-4,point[0]+4,point[1]+4),fill=color)
                d.text((left+7,450),f"family={(best or {}).get('family','?')[:30]}",fill=(210,220,231))
                d.text((left+7,473),f"waypoints={len(wp)} | base 2D only",fill=(239,177,95))
            else:
                d.text((left+12,top+60),'NO VALID 2D ROUTE',fill=(238,160,150))
                d.text((left+12,top+85),'Unknown / blocked / pre-handoff',fill=(216,220,227))
            d.text((left+7,510),'Neither IK nor garment clearance proven',fill=(232,125,116))
            d.text((left+7,533),'V2 action: ZERO after handoff',fill=(216,220,227))
            out=self.root/f'frame_{self.count:06d}.jpg' 
            canvas.save(out,format='JPEG',quality=self.quality,optimize=False)
            with self.index.open('a',newline='',encoding='utf-8') as f:
                csv.writer(f).writerow([self.count,frame,f'{t:.6f}',str(mode),int(holding),side,
                    '' if intent.get('yielding_robot') is None else intent['yielding_robot'],
                    '' if intent.get('passing_robot') is None else intent['passing_robot'],
                    intent.get('status','WAITING'),decision.get('status','WAITING'),
                    (decision.get('best') or {}).get('family','')])
            self.count+=1
        except Exception:
            self.errors+=1
            if self.errors==1: raise
