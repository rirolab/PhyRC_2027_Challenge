#!/usr/bin/env python3
"""Offline V4 audit: public pose trajectory + RGB-D 3D/2D evidence.

No simulator, robot action, collision certificate, or occupancy rewrite. Old
snapshots with no traversal history are shown as missing, never backfilled.
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
from diagnose_traversability import most_recent_run, get_latest_voxel_snapshot, load_snapshot, read_handoff_xy, choose_intent
from traversability_evidence import build_grounded_evidence, fuse_traversal_with_rgbd
from diagnose_ground_evidence import draw_panel


def read_traversal(path, target_shape):
    """Read ONLY stored sensor-history evidence (never synthesize poses)."""
    with np.load(path, allow_pickle=False) as d:
        keys=('base_pose_history','base_pose_history_frames',
              'traversed_centerline_count_by_robot_2d',
              'traversed_footprint_count_by_robot_2d')
        if not all(k in d.files for k in keys):
            return dict(history_present=False, pose_count=0, first_frame=None,
                        last_frame=None, centers=None, footprint=None,
                        discontinuities=0, reason='SNAPSHOT_HAS_NO_V4_TRAVERSAL_HISTORY')
        poses = np.asarray(d['base_pose_history'])
        frame = np.asarray(d['base_pose_history_frames'])
        centers = np.asarray(d['traversed_centerline_count_by_robot_2d'])
        footprint = np.asarray(d['traversed_footprint_count_by_robot_2d'])
        if poses.ndim!=3 or poses.shape[1:]!=(2,7) or frame.shape!=(len(poses),):
            raise ValueError('Invalid pose history shapes')
        if centers.shape!=(2,*target_shape) or footprint.shape!=centers.shape:
            raise ValueError('Stored traversal arrays do not align with map; refusing to reproject blindly')
        return dict(history_present=True, pose_count=len(poses),
                    first_frame=int(frame[0]) if len(frame) else None,
                    last_frame=int(frame[-1]) if len(frame) else None,
                    centers=centers, footprint=footprint,
                    discontinuities=int(d['traversal_discontinuities']) if 'traversal_discontinuities' in d.files else -1,
                    reason='PUBLIC_POSE_TRACE_ONLY_NOT_COLLISION_CERTIFIED')


def map_image(path, strict, grounded, review, fusion, origin, res, bases, route=None):
    shape=strict['base_cost_2d'].shape
    unobserved=np.full((*shape,3),(153,158,167),np.uint8)
    red=(220,80,74); green=(72,180,130); cyan=(57,186,212); purple=(162,97,189)
    s=unobserved.copy()
    s[strict['base_cost_2d']<255]=green
    s[review['base_review_obstacle_2d']]=red
    g=unobserved.copy()
    g[grounded['grounded_cost_2d']<255]=cyan
    g[strict['base_cost_2d']<255]=green
    g[review['base_review_obstacle_2d']]=red
    f=g.copy()
    f[fusion['traversal_observed_footprint_2d']]=[224,193,113]
    f[fusion['traversal_centerline_candidate_2d']]=purple
    f[grounded['grounded_cost_2d']<255]=cyan
    f[strict['base_cost_2d']<255]=green
    f[review['base_review_obstacle_2d']]=red
    panels=[draw_panel(im,origin,res,bases,title,route if title=='TRAVERSAL + RGBD' else None)
            for im,title in ((s,'STRICT'),(g,'GROUNDED'),(f,'TRAVERSAL + RGBD'))]
    margin, gap, header, footer=20,16,62,72
    wide=2*margin+sum(x.width for x in panels)+2*gap
    tall=header+max(x.height for x in panels)+footer
    im=Image.new('RGB',(wide,tall),'white');dr=ImageDraw.Draw(im)
    left=margin
    for name,panel in zip(('STRICT (fails closed)','RGB-D FLOOR + FREE RAY','POSE TRACE + RGB-D (UNVERIFIED)'),panels):
        dr.text((left,18),name,fill=(20,20,20))
        im.paste(panel,(left,header));left+=panel.width+gap
    dr.text((margin,tall-53),'PURPLE=visited centers | YELLOW=swept footprint | CYAN=RGB-D grounded | GREEN=strict | RED=obstacle',fill=(25,25,25))
    dr.text((margin,tall-30),'POSE TRACE IS NOT PROOF OF COLLISION-FREE TRAVEL. NO ACTIONS SENT. PLAN_ONLY.',fill=(156,35,35))
    im.save(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--logs',type=Path,default=Path('/output/v2_sdl'))
    ap.add_argument('--run',type=Path,default=None)
    ap.add_argument('--plan',action='store_true',help='Try an UNVERIFIED what-if route on fused map')
    ap.add_argument('--astar-expansions',type=int,default=650)
    args=ap.parse_args()
    run=args.run or most_recent_run(args.logs)
    snap=get_latest_voxel_snapshot(run)
    strict,review,origin,res,padded=load_snapshot(snap)
    ground=build_grounded_evidence(strict,review)
    trace=read_traversal(snap,strict['base_cost_2d'].shape)
    fusion=fuse_traversal_with_rgbd(strict,review,ground,
       centerline_count_by_robot=trace['centers'],
       footprint_count_by_robot=trace['footprint'])
    try:
        xy = read_handoff_xy(run)
        pose_ref = "HANDOFF"
    except RuntimeError as exc:
        if "Handoff positions missing" not in str(exc) or not trace["history_present"]:
            raise
        with np.load(snap, allow_pickle=False) as data:
            xy = np.asarray(data["base_pose_history"][-1, :, :2], dtype=float)
        if xy.shape != (2, 2) or not np.isfinite(xy).all():
            raise RuntimeError("Invalid latest base pose") from exc
        pose_ref = "LATEST_RECORDED_POSE"

    print("BASE_POSE_REFERENCE:", pose_ref)
    out=run/'perception_review';out.mkdir(parents=True,exist_ok=True)
    cost=fusion['traversal_fused_cost_2d']
    stats=dict(snapshot=str(snap),run_id=run.name,history_present=trace['history_present'],
        pose_samples=trace['pose_count'],history_first_frame=trace['first_frame'],
        history_last_frame=trace['last_frame'],pose_trace_reason=trace['reason'],
        discontinuities=trace['discontinuities'],
        strict_cells=int((strict['base_cost_2d']<255).sum()),
        rgbd_grounded_candidates=int((ground['grounded_cost_2d']<255).sum()),
        traversed_centerline_cells=int(fusion['traversal_observed_centerline_2d'].sum()),
        swept_footprint_cells=int(fusion['traversal_observed_footprint_2d'].sum()),
        trace_rgbd_conflicts=int(fusion['traversal_blocked_by_rgbd_2d'].sum()),
        accepted_trace_centers=int(fusion['traversal_centerline_candidate_2d'].sum()),
        fused_unverified=int(fusion['traversal_unverified_2d'].sum()),
        fused_candidates=int((cost<255).sum()),
        safety_status='PLAN_ONLY_UNVERIFIED_PUBLIC_POSE_TRACE_NOT_COLLISION_CERTIFIED',
        starts=[],planning=None)
    stats['base_pose_reference'] = pose_ref
    for robot, pt in enumerate(xy):
        ij=np.floor((pt-origin)/res).astype(int)
        inside=bool(np.all(ij>=0)&np.all(ij<cost.shape))
        key=tuple(ij) if inside else None
        stats['starts'].append(dict(robot=robot,xy=pt.tolist(),
            inside_map=inside,
            strict_free=bool(strict['base_cost_2d'][key]<255) if inside else False,
            traced_center=bool(fusion['traversal_observed_centerline_2d'][key]) if inside else False,
            rgbd_conflict=bool(fusion['traversal_blocked_by_rgbd_2d'][key]) if inside else None,
            fused_candidate=bool(cost[key]<255) if inside else False))
    route=None
    if args.plan and trace['history_present']:
        from strategy_decision import StrategyDecisionLayer
        planner=StrategyDecisionLayer(a_star_max_expansions=args.astar_expansions,rrt_iterations=12,max_trials=2)
        decision=planner.decide(cost,origin,res,xy,experience_library=[],conditional_intent=choose_intent(run))
        best=decision.get('best');route=best['waypoints'] if best else None
        stats['planning']=dict(status=decision.get('status'),waypoints=len(route) if route is not None else 0,
                               candidate_reasons=[x.get('reason') for x in decision.get('candidates',[])])
    elif args.plan:
        stats['planning']=dict(status='SKIPPED_NO_V4_HISTORY',waypoints=0)
    dest=out/'traversal_fusion_audit.json'
    dest.write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'traversal_fusion_map.npz',
        strict_cost_2d=strict['base_cost_2d'],
        grounded_cost_2d=ground['grounded_cost_2d'],
        fused_review_cost_2d=cost,
        observed_centerline_2d=fusion['traversal_observed_centerline_2d'],
        observed_footprint_2d=fusion['traversal_observed_footprint_2d'],
        observed_rgbd_conflicts_2d=fusion['traversal_blocked_by_rgbd_2d'],
        origin=origin,resolution=res,base_xy=xy)
    png=out/'traversal_fusion_compare.png'
    map_image(png,strict,ground,review,fusion,origin,res,xy,route)
    print('SOURCE:',snap)
    print('POSE_HISTORY:',trace['history_present'],'POSE_SAMPLES:',trace['pose_count'],
          'FIRST_LAST:',trace['first_frame'],trace['last_frame'],
          'DISCONTINUITIES:',trace['discontinuities'])
    print('STRICT:',stats['strict_cells'],'GROUNDED:',stats['rgbd_grounded_candidates'],
          'FUSED_REVIEW:',stats['fused_candidates'])
    print('TRACED_CENTERS:',stats['traversed_centerline_cells'],
          'SWEPT_FOOTPRINT:',stats['swept_footprint_cells'],
          'BLOCKED_BY_RGBD:',stats['trace_rgbd_conflicts'])
    for x in stats['starts']:
        print(f"R{x['robot']}: strict_free={x['strict_free']} traced={x['traced_center']} "
              f"fused_candidate={x['fused_candidate']} rgbd_conflict={x['rgbd_conflict']}")
    if stats['planning']:
        print('WHAT_IF:',stats['planning']['status'],'WAYPOINTS:',stats['planning']['waypoints'])
    if not trace['history_present']:
        print('OLD SNAPSHOT: V4 pose history absent. Cannot recreate unseen robot movement from RGB-D voxels.')
    print('NO ACTIONS SENT. NO MOTION CERTIFICATION.')
    print('PNG:',png)
    print('JSON:',dest)

if __name__=='__main__':main()
