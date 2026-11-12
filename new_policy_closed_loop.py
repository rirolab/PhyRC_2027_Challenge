"""PhyRC steps 1+2: reverse-roll hover followed by high-air staging.

Only public observations; never closes fingers, lowers to contact or lifts cloth.
Runtime files: this adapter, rgbd_servo.py, static_robot_camera.json, step3_shell.py.
"""
import os
import struct
import sys
import time
import zlib
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from rgbd_servo import RGBDServo
from step3_shell import Step3Shell

SNAP=True
SNAP_DIR='/output/anh'
SNAP_EVERY=5.
BRANCH_SNAP_COOLDOWN=0.5


def masks(img):
    """Original RGB filter retained; servo consistently augments pale yellow."""
    r,g,b=[img[...,i].astype(int) for i in range(3)]
    return (b>r+60)&(b>g+60),(r>170)&(g>170)&(b<110)


def save_png(path,img):
    img=np.ascontiguousarray(img[...,:3],dtype=np.uint8)
    h,w=img.shape[:2]
    raw=b''.join(b'\x00'+img[y].tobytes() for y in range(h))
    def chunk(tag,data):
        content=tag+data
        return struct.pack('>I',len(data))+content+struct.pack('>I',zlib.crc32(content)&0xffffffff)
    with open(path,'wb') as stream:
        stream.write(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+
                     chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))


def mark(img,x,y,color,size=5):
    h,w=img.shape[:2];x,y=int(round(x)),int(round(y))
    for d in (-1,0,1):
        if 0<=y+d<h:img[y+d,max(0,x-size):min(w,x+size+1)]=color
        if 0<=x+d<w:img[max(0,y-size):min(h,y+size+1),x+d]=color


def load_policy(checkpoint,observation_space,action_space):
    if checkpoint is not None:
        raise ValueError('Steps1+2 use public geometry, not a regression checkpoint')
    class Policy:
        def __init__(self):self.reset()

        def reset(self):
            self.dir=None;self.snap_t=-np.inf;self.error_log_t=-np.inf
            self.branch_losses=set()
            self.step3_shell=Step3Shell()
            self.branch_snap_t=-np.inf
            self.rgbd_servo=RGBDServo(masks,save_png,SNAP_DIR,SNAP_EVERY,SNAP)
            print('[POLICY] '+self.rgbd_servo.VERSION,flush=True)
            print('[STEP3] '+Step3Shell.VERSION+' descent_enabled=false',flush=True)

        def snap(self,obs,tags):
            if not SNAP:return
            try:
                folder=self.rgbd_servo.folder(self)
                images=[np.array(x,dtype=np.uint8,copy=True) for x in obs['rgb']]
                self.rgbd_servo.annotate(images,mark)
                sep=np.full((256,4,3),255,np.uint8)
                parts=[]
                for i,im in enumerate(images):
                    if i:parts.append(sep)
                    parts.append(im)
                t=float(np.asarray(obs['simulation_time_s']).ravel()[0])
                name=f't{t:06.1f}_{"__".join(tags)}.png'
                save_png(str(folder/name),np.concatenate(parts,axis=1))
                print('[ANH] '+name,flush=True)
            except (OSError,ValueError,KeyError) as exc:
                print('[ANH] save failed: '+repr(exc),flush=True)

        def __call__(self,obs):
            try:
                servo=self.rgbd_servo
                action=servo.step(obs)
                t=servo.t
                status=self.step3_shell.observe_servo(servo)
                servo.diag['step3']=status
                if status['event']:
                    servo.events.append(status['previous_phase']+'_TO_'+status['phase'])
                    print('[STEP3] phase='+status['phase']+' stop_reason='+str(status['stop_reason']),flush=True)
                missing=servo.diag.get('align_evidence',{}).get('missing_patch_robots',[])
                losses=set(missing)
                new_losses=losses-self.branch_losses
                self.branch_losses=losses
                capture_loss=bool(SNAP and new_losses and t-self.branch_snap_t>=BRANCH_SNAP_COOLDOWN)
                if capture_loss:
                    servo.events.extend(f'STAGING_VIEW_LOST_R{r}' for r in sorted(new_losses))
                    self.branch_snap_t=t
                servo.finalize(self,capture_depth=capture_loss)
                tags=servo.events.copy()
                if not tags and t-self.snap_t>=SNAP_EVERY:tags=[servo.phase]
                if tags:
                    self.snap_t=t;self.snap(obs,tags)
                return np.asarray(action,dtype=np.float32).reshape(2,9)
            except (ValueError,KeyError,np.linalg.LinAlgError) as exc:
                self.rgbd_servo.stop('EXCEPTION_'+repr(exc))
                print('[RGBD] EXCEPTION_BRAKE '+repr(exc),flush=True)
                return np.zeros((2,9),np.float32)
    return Policy()
