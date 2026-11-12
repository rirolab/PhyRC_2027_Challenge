#!/usr/bin/env python3
"""Encode all online observation montages into MP4 on Ubuntu VM host.

Policy automatically writes video_frames/frame_*.jpg and frame_index.csv. No
Isaac dependency here; ffmpeg must exist on host. Full run -> one MP4 per seed.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


def encode_one(run, *, fps=5., crf=23):
    folder=run/'video_frames'
    csv_path=folder/'frame_index.csv'
    if not csv_path.is_file():
        return dict(run=run.name,status='NO_VIDEO_FRAMES',frames=0)
    with csv_path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        return dict(run=run.name,status='EMPTY_VIDEO_FRAMES',frames=0)
    jpgs=sorted(folder.glob('frame_*.jpg'))
    if len(rows)!=len(jpgs) or any(not (folder/f'frame_{i:06d}.jpg').is_file() for i in range(len(rows))):
        return dict(run=run.name,status='FRAME_SEQUENCE_INCOMPLETE',frames=len(jpgs),index_rows=len(rows))
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        return dict(run=run.name,status='FFMPEG_NOT_INSTALLED',frames=len(jpgs))
    target=run/'v2_plan_only_5cam.mp4'
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-y',
         '-framerate',f'{fps:.6g}','-start_number','0',
         '-i',str(folder/'frame_%06d.jpg'),
         '-c:v','libx264','-preset','veryfast','-crf',str(int(crf)),
         '-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    try:
        subprocess.run(cmd,check=True,capture_output=True,text=True)
    except subprocess.CalledProcessError as e:
        return dict(run=run.name,status='ENCODE_FAILED',error=e.stderr[-1000:])
    return dict(run=run.name,status='MP4_OK',video=str(target),frames=len(rows),
                duration_s=len(rows)/float(fps),index=str(csv_path))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--logs',type=Path,default=Path('output/v2_sdl'))
    p.add_argument('--fps',type=float,default=5)
    p.add_argument('--crf',type=int,default=23)
    args=p.parse_args()
    if args.fps<=0:p.error('fps must be positive')
    runs=sorted(p.parent for p in args.logs.glob('*/video_frames'))
    if not runs:
        p.error('No video_frames directories; policy may not have run or video_enabled is false')
    results=[encode_one(run,fps=args.fps,crf=args.crf) for run in runs]
    successful=[r for r in results if r['status']=='MP4_OK']
    if successful:
        # Stable path simplifies gcloud compute scp from Windows CMD.
        latest_path=args.logs/'latest_plan_only_5cam.mp4'
        shutil.copy2(successful[-1]['video'],latest_path)
        print('Latest video:',latest_path)
    path=args.logs/'video_manifest.json'
    path.write_text(json.dumps(results,indent=2,ensure_ascii=False))
    for r in results:print(json.dumps(r,ensure_ascii=False))
    print('Manifest:',path)
    if not any(r['status']=='MP4_OK' for r in results):raise SystemExit(2)

if __name__=='__main__':main()
