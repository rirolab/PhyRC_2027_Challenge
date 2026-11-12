#!/usr/bin/env python3
"""Offline comparison of strict vs UNVERIFIED review traversability.

Reads ONLY public-observation-derived snapshots; never connects to the sim or
executes a robot policy. For old 7x7m snapshots it adds an UNOBSERVED 1m band
west of the map to diagnose the clipped R0 starting footprint. The band is
NEVER certified free. Designed to run with ./run.sh python ...bootstrap...
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "participant"))
from costmap_memory import GridSpec, OccupancyMemory


def most_recent_run(logs):
    paths=list(logs.glob('*/strategy_decisions.jsonl'))
    if not paths:
        raise FileNotFoundError('No strategy_decisions.jsonl under '+str(logs))
    return max(paths,key=lambda p:p.stat().st_mtime).parent


def get_latest_voxel_snapshot(run):
    snapshots=[p for p in run.glob('*_full.npz') if p.is_file()]
    if not snapshots:
        raise FileNotFoundError('No *_full.npz voxel snapshot: run has not saved raw map')
    return max(snapshots,key=lambda p:p.stat().st_mtime)


def read_handoff_xy(run):
    report=run/'report.json'
    if report.exists():
        obj=json.loads(report.read_text())
        poses=(obj.get('handoff') or {}).get('base_pose_world')
        if poses is not None:
            xy=np.asarray(poses,float)[:,:2]
            if xy.shape==(2,2) and np.isfinite(xy).all():return xy
    raise RuntimeError('Handoff positions missing: inspect run/report.json')


def load_snapshot(path, *, expand_legacy=True):
    with np.load(path,allow_pickle=False) as raw:
        needed=('voxel_log_odds','voxel_observation_count','voxel_camera_mask',
                'voxel_view_sector_mask','voxel_origin','voxel_resolution',
                'floor_subcells','low_returns')
        missing=[k for k in needed if k not in raw]
        if missing:raise ValueError('Missing arrays in '+str(path)+': '+repr(missing))
        lod=np.asarray(raw['voxel_log_odds'],np.float32)
        nobs=np.asarray(raw['voxel_observation_count'],np.uint32)
        cam=np.asarray(raw['voxel_camera_mask'],np.uint8)
        sec=np.asarray(raw['voxel_view_sector_mask'],np.uint32)
        origin=np.asarray(raw['voxel_origin'],float)
        resolution=float(raw['voxel_resolution'])
        floor=np.asarray(raw['floor_subcells'],np.uint32)
        low=np.asarray(raw['low_returns'],bool)
    if lod.ndim!=3 or lod.shape!=nobs.shape or origin.shape!=(3,):
        raise ValueError('Invalid voxel snapshot shapes')
    if floor.shape!=(lod.shape[0]*4,lod.shape[1]*4) or low.shape!=(lod.shape[0]*2,lod.shape[1]*2):
        raise ValueError('Expected 2.5cm floor subcells and 5cm low obstacles for 10cm voxel grid')
    # Previously the 7m map cut R0's 0.235m footprint at x=-3.5.
    # Add cells with ZERO measurements, as a diagnostic what-if only.
    padded=False
    if expand_legacy and origin[0]>-4.0:
        width=int(round(1.0/resolution))
        pad=((width,0),(0,0),(0,0))
        lod=np.pad(lod,pad)
        nobs=np.pad(nobs,pad)
        cam=np.pad(cam,pad)
        sec=np.pad(sec,pad)
        floor=np.pad(floor,((4*width,0),(0,0)))
        low=np.pad(low,((2*width,0),(0,0)))
        origin[0]-=width*resolution
        padded=True
    spec=GridSpec(origin=tuple(origin),resolution=resolution,shape=lod.shape)
    vol=OccupancyMemory(spec)
    vol.log_odds=lod;vol.observation_count=nobs
    vol.camera_mask=cam;vol.view_sector_mask=sec
    floor_cube=floor.reshape(lod.shape[0]*2,2,lod.shape[1]*2,2)
    votes=floor_cube.sum(axis=(1,3))
    # Same strict support as mapper.py.
    support=(floor_cube.min(axis=(1,3))>=2)&~low
    strict=vol.project_base(support,output_res=.05)
    review=vol.project_base_review(strict,floor_votes=votes,
                                   near_floor_obstacle=low,output_res=.05)
    return strict,review,origin[:2],.05,padded


def cell_status(map_array, origin, resolution, xy):
    cell=np.floor((np.asarray(xy)-origin)/resolution).astype(int)
    inside=bool(np.all(cell>=0) and np.all(cell<map_array.shape))
    return dict(cell=cell.tolist(),inside_map=inside,
                cost=int(map_array[tuple(cell)]) if inside else None,
                traversable=bool(map_array[tuple(cell)]<255) if inside else False)


def draw_comparison(path, strict, review, origin, resolution, poses):
    strict_cost=strict['base_cost_2d']
    review_cost=review['base_review_cost_2d']
    occupancy=review['base_review_obstacle_2d']
    hypothesis=review['base_review_hypothesis_only_2d']
    # Indexed as [x,y]. Render x right, positive y up.
    left=np.full((*strict_cost.shape,3),[150,155,160],np.uint8)
    left[strict_cost<255]=[100,195,145]
    left[strict['base_occupied_2d']]=[224,85,83]
    right=np.full((*review_cost.shape,3),[150,155,160],np.uint8)
    right[hypothesis]=[234,174,64]
    right[review['base_review_floor_votes_2d']>0]=[102,175,215]
    right[(review_cost<255)&(~hypothesis)]=[100,195,145]
    right[occupancy]=[224,85,83]
    def tile(arr):
        # array x/y axes -> image row (reversed y), col (x)
        ar=np.transpose(arr,(1,0,2))[::-1]
        im=Image.fromarray(ar,'RGB').resize((ar.shape[1]*4,ar.shape[0]*4),Image.Resampling.NEAREST)
        dr=ImageDraw.Draw(im)
        for i,xy in enumerate(poses):
            x=int((xy[0]-origin[0])/resolution*4)
            y=int((ar.shape[0]-(xy[1]-origin[1])/resolution)*4)
            dr.ellipse((x-8,y-8,x+8,y+8),fill='black',outline='white',width=3)
            dr.text((x+9,y-12),f'R{i}',fill='black',stroke_width=1,stroke_fill='white')
        return im
    l,r=tile(left),tile(right)
    padding=30;header=62;footer=70
    canvas=Image.new('RGB',(padding*3+l.width+r.width,header+max(l.height,r.height)+footer),'white')
    canvas.paste(l,(padding,header));canvas.paste(r,(padding*2+l.width,header))
    d=ImageDraw.Draw(canvas)
    d.text((padding,15),'STRICT: green=certified base-map cell | grey=unknown',fill=(20,20,20))
    d.text((padding*2+l.width,15),'REVIEW: amber=hypothesis ONLY | blue=floor votes',fill=(20,20,20))
    d.text((padding,header+l.height+12),'RED=observed obstacle; black circles=handoff bases; UNKNOWN never certified FREE.',fill=(30,30,30))
    d.text((padding,header+l.height+35),'PLAN_ONLY / WHAT-IF; not collision-certified and not authorized for execution.',fill=(150,30,30))
    canvas.save(path)


def choose_intent(run):
    log=run/'strategy_decisions.jsonl'
    intent={}
    if not log.exists():return intent
    # Keep only causal role choice from THIS run, not older episodes.
    for line in log.read_text().splitlines():
        try: obj=json.loads(line)
        except json.JSONDecodeError:continue
        if obj.get('event')=='SCENE_ROLE_AUTO_SELECTED':
            x=obj.get('intent')
            if isinstance(x,dict) and x.get('role_selected') and x.get('strategy_id'):
                intent=x
    return intent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--logs',type=Path,default=Path('/output/v2_sdl'))
    ap.add_argument('--run',type=Path,default=None,help='specific run folder')
    ap.add_argument('--plan',action='store_true',help='bounded hypothetical coupled planner; slower')
    ap.add_argument('--astar-expansions',type=int,default=350)
    args=ap.parse_args()
    run=args.run if args.run is not None else most_recent_run(args.logs)
    snapshot=get_latest_voxel_snapshot(run)
    xy=read_handoff_xy(run)
    strict,review,origin,res,padded=load_snapshot(snapshot)
    strict_cost=strict['base_cost_2d'];review_cost=review['base_review_cost_2d']
    starts=[]
    for i in (0,1):
        starts.append(dict(robot=i,xy=xy[i].tolist(),
             strict=cell_status(strict_cost,origin,res,xy[i]),
             review=cell_status(review_cost,origin,res,xy[i])))
    summary=dict(run=run.name,source_snapshot=snapshot.name,origin=origin.tolist(),
         resolution_m=res,grid_shape=list(review_cost.shape),
         legacy_unobserved_padding_added=padded,
         strict_traversable_cells=int(np.count_nonzero(strict_cost<255)),
         review_candidate_cells=int(np.count_nonzero(review_cost<255)),
         review_unverified_cells=int(review['base_review_hypothesis_only_2d'].sum()),
         measured_floor_cells=int((review['base_review_floor_votes_2d']>0).sum()),
         obstacle_cells=int(review['base_review_obstacle_2d'].sum()),
         start=starts,mode='OFFLINE_UNVERIFIED_REVIEW_NOT_FOR_EXECUTION',
         planning=None)
    out=run/'perception_review';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'traversability_review_map.npz',
         strict_cost_2d=strict_cost,review_cost_2d=review_cost,
         observed_obstacle_2d=review['base_review_obstacle_2d'],
         hypothesis_only_2d=review['base_review_hypothesis_only_2d'],
         floor_votes_2d=review['base_review_floor_votes_2d'],
         observed_free_2d=review['base_review_observed_free_2d'],
         origin=origin,resolution=res,base_xy=xy)
    draw_comparison(out/'traversability_compare.png',strict,review,origin,res,xy)
    if args.plan:
        from strategy_decision import StrategyDecisionLayer
        # Never invoke PairedBaseExecutor. Review costs cannot certify motion.
        planner=StrategyDecisionLayer(a_star_max_expansions=max(25,args.astar_expansions),
                    rrt_iterations=12,max_trials=2)
        result=planner.decide(review_cost,origin,res,xy,
                    experience_library=[],conditional_intent=choose_intent(run))
        best=result.get('best')
        summary['planning']=dict(status=result.get('status'),
            candidates=[dict(family=v.get('family'),status=v.get('status'),
                             reason=v.get('reason'),astar_expansions=v.get('astar_expansions'))
                        for v in result.get('candidates',[])],
            base_waypoint_count=len(best.get('waypoints',[])) if best else 0,
            full_robot_execution_authorized=False,
            safety_status='UNVERIFIED_BASE_HYPOTHESIS_UNKNOWN_CELLS_ALLOWED')
    (out/'traversability_review.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('SOURCE:',snapshot)
    print('STRICT traversable:',summary['strict_traversable_cells'])
    print('REVIEW candidates:',summary['review_candidate_cells'],
          '(unverified:',summary['review_unverified_cells'],')')
    print('Observed floor:',summary['measured_floor_cells'])
    for item in starts:
        print(f"R{item['robot']}: strict={item['strict']['traversable']} review={item['review']['traversable']} "
              f"strict_cost={item['strict']['cost']} review_cost={item['review']['cost']}")
    if args.plan:
        print('WHAT-IF PLANNER:',summary['planning']['status'],
              'waypoints:',summary['planning']['base_waypoint_count'])
        print('NOT MOTION CERTIFIED. NO ACTIONS SENT.')
    print('PNG:',out/'traversability_compare.png')
    print('JSON:',out/'traversability_review.json')

if __name__=='__main__':main()
