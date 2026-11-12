#!/usr/bin/env python3
"""Visualize PhyRC Garment Observation State v0 on exact recorded frames.

Uses only actor RGB/depth observations and existing v0 frame JSONL, plus public
camera intrinsics. Read-only: no policy changes, no audit or privileged state.
Shows selected camera, each actual RGB image, HSV mask and sparse depth points.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _common import CAMERAS, get_camera_arrays, read_depth_frame
from reconstruct_sparse_rgbd import DEFAULT_HFOV, camera_matrix, sparse_points
from phase2_reorientation.perception import CameraFrame, MultiHSVRangeSegmenter, PerceptionProcessor

CLIP_MAX = {'overview':10., 'robot_0_wrist':3., 'robot_1_wrist':3.,
            'robot_0_head':5., 'robot_1_head':5.}
TILE_W, TILE_H = 256, 256


def label_color(selected: bool):
    return (42, 128, 69) if selected else (68, 74, 85)


def compose_panel(rgb, mask, uv, xyz):
    original = Image.fromarray(rgb.astype(np.uint8), 'RGB')
    overlay = np.array(original).copy()
    overlay[mask] = np.rint(.65 * overlay[mask] + .35 * np.array([20, 230, 80])).astype(np.uint8)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), 'L').convert('RGB')
    dot_img = original.copy()
    draw = ImageDraw.Draw(dot_img)
    if len(uv):
        zs = np.asarray(xyz[:, 2], dtype=float)
        lo, hi = np.quantile(zs, [.05, .95])
        t = (zs - lo) / max(hi - lo, 1e-8)
        for (u, v), zt in zip(uv, t):
            c = (int(255 * zt), int(70 + 160 * (1 - abs(2 * zt - 1))), int(255 * (1 - zt)))
            draw.point((int(u), int(v)), fill=c)
    return original, Image.fromarray(overlay, 'RGB'), mask_img, dot_img


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5',type=Path,required=True)
    parser.add_argument('--states',type=Path,required=True,help='garment_observation_frames.jsonl from v0')
    parser.add_argument('--calibration',type=Path,action='append',required=True)
    parser.add_argument('--demo',default='demo_0')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='*',help='Only these frame IDs, which must exist in v0 states')
    parser.add_argument('--min-component-pixels',type=int,default=80)
    parser.add_argument('--keep-components',type=int,default=3)
    parser.add_argument('--min-mask-pixels',type=int,default=30)
    parser.add_argument('--max-points',type=int,default=2048)
    a=parser.parse_args()
    if len(a.calibration)!=2: parser.error('Pass exactly two --calibration')
    if a.max_points<1: parser.error('--max-points must be positive')
    states={}
    for line in a.states.read_text(encoding='utf8').splitlines():
        if line.strip():
            obj=json.loads(line)
            states[int(obj['frame'])]=obj
    selected_ids=sorted(states)
    if a.frames:
        missing=sorted(set(a.frames)-set(states))
        if missing: parser.error(f'Frames not present in v0 states: {missing}. Use saved frame IDs; do not infer selection from neighboring frames.')
        selected_ids=sorted(set(a.frames))
    if not selected_ids: parser.error('No selected frame IDs')
    bounds=[]
    for fp in a.calibration:
        d=json.loads(fp.read_text(encoding='utf8'))
        bounds.append((tuple(d['hsv_lower']),tuple(d['hsv_upper'])))
    segmenter=MultiHSVRangeSegmenter(hsv_ranges=tuple(bounds),min_component_pixels=a.min_component_pixels,
                                     keep_largest_components=a.keep_components)
    processor=PerceptionProcessor(segmenter=segmenter,min_mask_pixels=a.min_mask_pixels)
    a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    font=ImageFont.load_default()
    header_h=80
    one_camera_w=256
    one_camera_h=header_h+256*2+30
    with h5py.File(a.hdf5,'r') as f:
        obs=f[f'data/{a.demo}/obs']
        src={name:get_camera_arrays(obs,name) for name in CAMERAS}
        for fi in selected_ids:
            frame_state=states[fi]
            picked=frame_state['selected_observation_camera']
            state_per_cam={x['camera']:x for x in frame_state['cameras']}
            sheet=Image.new('RGB',(5*one_camera_w,one_camera_h+56),(239,242,245))
            draw=ImageDraw.Draw(sheet)
            draw.text((12,10),f'Frame {fi} | Selected: {picked or "NONE"} | Green border = selected',fill=(25,30,38),font=font)
            draw.text((12,28),'Top: RGB | Middle: HSV mask on RGB | Bottom: binary mask / sparse XYZ dots',fill=(50,58,66),font=font)
            for ci,cam in enumerate(CAMERAS):
                ar,ad,av=src[cam]
                rgb=np.asarray(ar[fi],dtype=np.uint8)
                depth=read_depth_frame(ad,fi).astype(np.float32)
                valid=read_depth_frame(av,fi).astype(bool)
                det=processor.process_frame(CameraFrame(camera_name=cam,rgb=rgb,depth_m=depth,depth_valid=valid))
                mask=np.asarray(det.mask,dtype=bool)
                K=camera_matrix(rgb.shape[1],rgb.shape[0],DEFAULT_HFOV[cam])
                xyz,uv,available=sparse_points(mask,depth,valid,K,a.max_points,0.,CLIP_MAX[cam])
                st=state_per_cam[cam]
                is_selected=(cam==picked)
                x0=ci*one_camera_w
                col=label_color(is_selected)
                if is_selected: draw.rectangle((x0,54,x0+one_camera_w-1,one_camera_h+55),outline=(25,165,75),width=4)
                draw.rectangle((x0+3,58,x0+252,112),fill=col)
                draw.text((x0+9,64),cam + ('  SELECTED' if is_selected else ''),font=font,fill=(255,255,255))
                draw.text((x0+9,78),f"q={st['quality_heuristic']:.3f} | status={st['status']}",font=font,fill=(255,255,255))
                draw.text((x0+9,94),f"mask={int(mask.sum())} px | xyz={len(xyz)} | edge={int(st['mask_edge_touch'])}",font=font,fill=(255,255,255))
                original,blend,mask_image,dots=compose_panel(rgb,mask,uv,xyz)
                # Four distinct views arranged 2x2 at half size, preserving full 256x256 sheet column
                thumb=128
                for im,dx,dy in [(original,0,0),(blend,128,0),(mask_image,0,128),(dots,128,128)]:
                    sheet.paste(im.resize((thumb,thumb),Image.Resampling.NEAREST),(x0+dx,118+dy))
                sheet.paste(blend,(x0,380))
                draw.text((x0+8,640),'Large: RGB + HSV mask (green)',fill=(30,40,50),font=font)
                records.append({'frame':fi,'camera':cam,'selected':is_selected,
                                'v0_score':st['quality_heuristic'],'v0_mask_pixels':st['mask_area_px'],
                                'recomputed_mask_pixels':int(mask.sum()),'v0_xyz_points':st['xyz_points'],
                                'recomputed_xyz_points':int(len(xyz)),
                                'matching_mask':bool(st['mask_area_px']==int(mask.sum())),
                                'matching_xyz':bool(st['xyz_points']==int(len(xyz)))})
            path=a.output/f'frame_{fi:06d}_selection.png'
            sheet.save(path,optimize=True)
            print(f'frame={fi} selected={picked or "NONE"} saved={path.name}')
    with (a.output/'visualization_checks.csv').open('w',newline='',encoding='utf8') as outf:
        writer=csv.DictWriter(outf,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (a.output/'README.txt').write_text(
        'Each image shows all 5 cameras for an EXACT frame recorded by Garment Observation State v0.\n'
        'Upper 2x2 patches per camera: original RGB, overlay of HSV mask, binary HSV mask, sparse XYZ depth-colored points.\n'
        'Lower large image: RGB + HSV mask overlay. Green outline means v0-selected camera.\n'
        'Scores are heuristic and not semantic garment confidence.\n'
        'Check visualization_checks.csv: mismatches mean config/code differs from original run.\n'
        'No cross-camera XYZ fusion or world-camera extrinsic is used.\n',encoding='utf8')
    print(f'Done: {len(selected_ids)} frames, {len(records)} camera observations. Output: {a.output}')

if __name__=='__main__': main()
