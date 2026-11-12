#!/usr/bin/env python3
"""Audit RGB-D floor/free-ray/voxel-height projection; no Isaac execution.

Run inside the existing Isaac Docker via v2_python_bootstrap.py. Handles legacy
7x7 snapshots without turning padded space into an observation.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'participant'))
sys.path.insert(0,str(ROOT/'v2_tools'))
from diagnose_traversability import (most_recent_run, get_latest_voxel_snapshot,
                                     read_handoff_xy, load_snapshot, choose_intent)
from traversability_evidence import (build_grounded_evidence, start_causal_audit,
                                     path_evidence_audit)


def snapshot_details(path, expected_shape):
    with np.load(path,allow_pickle=False) as d:
        lod=np.asarray(d['voxel_log_odds'],np.float32)
        nobs=np.asarray(d['voxel_observation_count'],np.uint32)
        origin=np.asarray(d['voxel_origin'],float)
        resolution=float(d['voxel_resolution'])
        floor=np.asarray(d['floor_subcells'],np.uint32)
        visits=np.asarray(d['base_visit_count_2d'],np.uint16) if 'base_visit_count_2d' in d else None
    # New maps are 8m wide. Old 7m maps need a one-meter UNOBSERVED left pad.
    padx=expected_shape[0]*2-floor.shape[0]
    if padx<0 or expected_shape[1]*2!=floor.shape[1] or padx%4:
        raise ValueError('snapshot and strict grid have incompatible dimensions')
    if padx:
        floor=np.pad(floor,((padx,0),(0,0)),constant_values=0)
        if visits is not None:visits=np.pad(visits,((padx//2,0),(0,0)),constant_values=0)
    cube=floor.reshape(expected_shape[0],2,expected_shape[1],2)
    min_subcell=cube.min(axis=(1,3)).astype(np.uint16)
    z=origin[2]+(np.arange(lod.shape[2])+.5)*resolution
    band=(z>=.10)&(z<=.65)
    if not band.any():raise ValueError('no voxels in base height band')
    prob=1/(1+np.exp(-lod))
    unknown=(nobs==0)|((prob>.33)&(prob<.67))
    unknown_band=unknown[:,:,band].sum(axis=2).astype(np.uint8)
    if padx:
        unknown_band=np.pad(unknown_band,((padx//4,0),(0,0)),constant_values=int(band.sum()))
    unknown_2d=np.repeat(np.repeat(unknown_band,2,axis=0),2,axis=1)
    return dict(strict_floor_min=min_subcell,height_unknown_count=unknown_2d,
                base_visits=visits,base_height_voxel_count=int(band.sum()),
                history_present=visits is not None)


def draw_panel(grid, origin, resolution, starts, title, route=None, *, step=4):
    arr=np.asarray(grid,np.uint8)
    if arr.ndim!=3 or arr.shape[2]!=3:raise ValueError('RGB panel required')
    view=Image.fromarray(np.transpose(arr,(1,0,2))[::-1],'RGB')
    view=view.resize((view.width*step,view.height*step),Image.Resampling.NEAREST)
    dr=ImageDraw.Draw(view)
    def point(pos):
        x=int(round((pos[0]-origin[0])/resolution*step))
        y=int(round((grid.shape[1]-(pos[1]-origin[1])/resolution)*step))
        return (x,y)
    if route is not None:
        p=np.asarray(route,float)
        if p.ndim==3 and p.shape[1:]==(2,2):
            for robot,color in ((0,(40,30,230)),(1,(200,30,160))):
                xy=[point(v) for v in p[:,robot,:]]
                if len(xy)>1:dr.line(xy,fill=color,width=3,joint='curve')
                for q in xy:dr.ellipse((q[0]-2,q[1]-2,q[0]+2,q[1]+2),fill=color)
    for i,p in enumerate(starts):
        x,y=point(p)
        dr.ellipse((x-7,y-7,x+7,y+7),fill='black',outline='white',width=2)
        dr.text((x+8,y-10),'R'+str(i),fill='black',stroke_width=1,stroke_fill='white')
    return view


def make_comparison(dst, strict, review, grounded, origin, res, xy, broad_waypoints=None):
    size=strict['base_cost_2d'].shape
    grey=(154,160,167); red=(224,85,83);green=(100,195,145)
    strict_rgb=np.full((*size,3),grey,np.uint8)
    strict_rgb[strict['base_cost_2d']<255]=green
    strict_rgb[strict['base_occupied_2d']]=red
    # Green = strict, cyan = measured floor+ray candidate; purple = interpolation
    # Amber = review-only hypothesis; red = observed obstacle. Distinct statuses.
    evidence_rgb=np.full((*size,3),grey,np.uint8)
    measured=grounded['grounded_measured_floor_2d']
    free_ray=grounded['grounded_ray_free_2d']
    evidence_rgb[measured]=[122,177,203]
    evidence_rgb[measured & free_ray]=[62,161,206]
    evidence_rgb[grounded['grounded_interpolated_floor_2d']]=[170,133,217]
    evidence_rgb[grounded['grounded_candidate_2d']]=[54,210,190]
    evidence_rgb[strict['base_cost_2d']<255]=green
    evidence_rgb[review['base_review_obstacle_2d']]=red
    broad_rgb=np.full((*size,3),grey,np.uint8)
    broad_rgb[review['base_review_hypothesis_only_2d']]=[234,174,64]
    broad_rgb[grounded['grounded_candidate_2d']]=[54,210,190]
    broad_rgb[strict['base_cost_2d']<255]=green
    broad_rgb[review['base_review_obstacle_2d']]=red
    panel=[draw_panel(strict_rgb,origin,res,xy,'STRICT'),
           draw_panel(evidence_rgb,origin,res,xy,'GROUNDED',broad_waypoints),
           draw_panel(broad_rgb,origin,res,xy,'BROAD',broad_waypoints)]
    margin=24; head=70;foot=82;gap=20
    width=margin*2+sum(p.width for p in panel)+gap*2
    height=head+max(p.height for p in panel)+foot
    image=Image.new('RGB',(width,height),'white')
    pen=ImageDraw.Draw(image)
    left=margin
    for name,p in zip(('STRICT (unknown denied)','FLOOR+RAY (review only)','BROAD (unknown hypothesis)'),panel):
        pen.text((left,20),name,fill=(15,15,15))
        image.paste(p,(left,head))
        left+=p.width+gap
    pen.text((margin,height-64),'CYAN=floor+free ray | PURPLE=interpolated floor | AMBER=unobserved hypothesis | RED=obstacle',fill=(24,24,24))
    pen.text((margin,height-42),'Route overlay: R0=blue R1=magenta; NOT motion certified; PLAN_ONLY; unknown 3D heights remain unknown.',fill=(156,37,37))
    image.save(dst)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--logs',type=Path,default=Path('/output/v2_sdl'))
    parser.add_argument('--run',type=Path,default=None)
    parser.add_argument('--plan',action='store_true',help='also audit an unverified paired candidate')
    parser.add_argument('--astar-expansions',type=int,default=200)
    args=parser.parse_args()
    run=args.run or most_recent_run(args.logs)
    snapshot=get_latest_voxel_snapshot(run)
    strict,review,origin,res,padded=load_snapshot(snapshot)
    xy=read_handoff_xy(run)
    details=snapshot_details(snapshot,strict['base_cost_2d'].shape)
    grounded=build_grounded_evidence(strict,review,base_visits=details['base_visits'])
    audit=start_causal_audit(xy,origin,res,strict,review,grounded,
        strict_floor_min=details['strict_floor_min'],
        height_unknown_count=details['height_unknown_count'])
    stats=dict(snapshot=str(snapshot),run=run.name,legacy_padding=padded,
        basis='PUBLIC_RGBD_PLUS_PUBLIC_BASE_POSE_HISTORY',
        strict_cells=int((strict['base_cost_2d']<255).sum()),
        broad_review_cells=int((review['base_review_cost_2d']<255).sum()),
        grounded_candidates=int(grounded['grounded_candidate_2d'].sum()),
        grounded_unverified=int(grounded['grounded_unverified_2d'].sum()),
        measured_floor_cells=int(grounded['grounded_measured_floor_2d'].sum()),
        interpolated_floor_cells=int(grounded['grounded_interpolated_floor_2d'].sum()),
        floor_plus_free_ray_cells=int((grounded['grounded_measured_floor_2d'] & grounded['grounded_ray_free_2d']).sum()),
        free_ray_cells=int(grounded['grounded_ray_free_2d'].sum()),
        base_height_voxel_count=details['base_height_voxel_count'],
        base_pose_history_present=details['history_present'],
        start_audit=audit,
        safety_scope='OBSERVATION_ONLY_NOT_COLLISION_CERTIFIED',planning=None)
    route=None
    if args.plan:
        from strategy_decision import StrategyDecisionLayer
        planner=StrategyDecisionLayer(a_star_max_expansions=max(25,args.astar_expansions),
                                      rrt_iterations=12,max_trials=2)
        role=choose_intent(run)
        broad=planner.decide(review['base_review_cost_2d'],origin,res,xy,
                            experience_library=[],conditional_intent=role)
        best=broad.get('best')
        if best:
            route=best['waypoints']
            stats['planning']=dict(broad_status=broad.get('status'),
                broad_waypoints=len(route),
                broad_route_evidence=path_evidence_audit(route,origin,res,strict,review,grounded))
        else:
            stats['planning']=dict(broad_status=broad.get('status'),broad_waypoints=0,
                broad_reasons=[x.get('reason') for x in broad.get('candidates',[])])
        # Deliberately do not execute or treat the broad route as an acceptable plan.
    out=run/'perception_review'
    out.mkdir(exist_ok=True)
    dest=out/'grounded_evidence_audit.json'
    dest.write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
    np.savez_compressed(out/'grounded_evidence_map.npz',
        strict_cost_2d=strict['base_cost_2d'],
        broad_review_cost_2d=review['base_review_cost_2d'],
        grounded_cost_2d=grounded['grounded_cost_2d'],
        measured_floor_2d=grounded['grounded_measured_floor_2d'],
        interpolated_floor_2d=grounded['grounded_interpolated_floor_2d'],
        measured_free_ray_2d=grounded['grounded_ray_free_2d'],
        observed_obstacle_2d=review['base_review_obstacle_2d'],
        base_visit_count_2d=grounded['grounded_base_visit_count_2d'],
        height_unknown_count_2d=details['height_unknown_count'],
        origin=origin,resolution=res,base_xy=xy)
    image=out/'grounded_evidence_compare.png'
    make_comparison(image,strict,review,grounded,origin,res,xy,broad_waypoints=route)
    print('SOURCE:',snapshot)
    print('STRICT:',stats['strict_cells'],'BROAD_REVIEW:',stats['broad_review_cells'],
          'EVIDENCE_GROUNDED:',stats['grounded_candidates'])
    print('FLOOR:',stats['measured_floor_cells'],'FLOOR_PLUS_FREE_RAY:',stats['floor_plus_free_ray_cells'])
    for row in audit:
        print('R'+str(row['robot']),'strict_cost=',row.get('strict_cost'),
              'grounded_cost=',row.get('grounded_cost'),'floor_votes=',row.get('floor_votes'),
              'free_height_layers=',row.get('free_height_layers'),
              'unknown_height_layers=',row.get('unknown_base_height_layers'),
              'reasons=',','.join(row['reasons']))
    if stats['planning']:
        print('BROAD_ROUTE:',stats['planning']['broad_status'],
              'waypoints:',stats['planning']['broad_waypoints'])
        if route:print('BROAD_ROUTE_GROUNDED_FRACTION:',stats['planning']['broad_route_evidence']['grounded_fraction'])
    print('NO ACTIONS SENT. NO UNKNOWN-FREE CERTIFICATION.')
    print('PNG:',image)
    print('JSON:',dest)

if __name__=='__main__':main()
