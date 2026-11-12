#!/usr/bin/env python3
"""PhyRC M1: observation-only robot geometry and planning-contract extraction.

Not a full collision checker, IK solver or pose estimator for garments/manikin.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import h5py


def rotate_wxyz(q, p):
    q=np.asarray(q,dtype=float); p=np.asarray(p,dtype=float)
    if q.shape!=(4,) or p.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError('invalid quaternion/position')
    n=float(np.linalg.norm(q))
    if n<1e-8 or abs(n-1)>0.05: raise ValueError(f'quaternion norm invalid: {n}')
    w,x,y,z=q/n
    v=np.array([x,y,z],dtype=float)
    return p + 2*w*np.cross(v,p) + 2*np.cross(v,np.cross(v,p))


def transform_point(base7, point3):
    return np.asarray(base7[:3],dtype=float)+rotate_wxyz(np.asarray(base7[3:7]), point3)


def check_snapshot(base, grasp, fingertips, distances, close, tol=2e-5):
    base=np.asarray(base,float).reshape(2,7)
    grasp=np.asarray(grasp,float).reshape(2,7)
    fingertips=np.asarray(fingertips,float).reshape(2,2,3)
    distances=np.asarray(distances,float).reshape(2)
    close=np.asarray(close,bool).reshape(2)
    checks=[]; robots=[]
    for i in range(2):
        geom=float(np.linalg.norm(fingertips[i,0]-fingertips[i,1]))
        err=abs(geom-distances[i])
        qnorm=float(np.linalg.norm(base[i,3:7])); gnorm=float(np.linalg.norm(grasp[i,3:7]))
        valid=(np.isfinite(base[i]).all() and np.isfinite(grasp[i]).all() and
               np.isfinite(fingertips[i]).all() and np.isfinite(distances[i]) and
               abs(qnorm-1)<.05 and abs(gnorm-1)<.05 and err<=tol)
        tip_world=[transform_point(base[i],v).tolist() for v in fingertips[i]] if valid else None
        grip_world=transform_point(base[i],grasp[i,:3]).tolist() if valid else None
        robots.append(dict(robot=i,base_pose_world=base[i].tolist(),grasp_pose_base=grasp[i].tolist(),
                           grasp_origin_world=grip_world,fingertips_base=fingertips[i].tolist(),
                           fingertips_world=tip_world,close_intent=bool(close[i]),
                           fingertip_distance_m=float(distances[i]),geometry_distance_m=geom,
                           distance_error_m=err,base_quaternion_norm=qnorm,
                           grasp_quaternion_norm=gnorm,geometry_valid=bool(valid)))
        checks.append(valid)
    base_sep=float(np.linalg.norm(base[0,:2]-base[1,:2]))
    grip_sep=(float(np.linalg.norm(np.array(robots[0]['grasp_origin_world'])-
                                   np.array(robots[1]['grasp_origin_world']))) if all(checks) else None)
    return dict(robots=robots,base_xy_separation_m=base_sep,
                grasp_origin_separation_world_m=grip_sep,geometry_valid=bool(all(checks)))


def interpret_proxy(separation, radius0, radius1, margin):
    if any(x<0 or not np.isfinite(x) for x in (separation,radius0,radius1,margin)):
        raise ValueError('invalid proxy')
    clearance=separation-radius0-radius1
    return dict(base_proxy_clearance_m=float(clearance),
                base_proxy_warning=bool(clearance<margin))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hdf5',type=Path,required=True)
    ap.add_argument('--demo',default='demo_0')
    ap.add_argument('--frames',type=int,nargs='+',default=[1700,1730,1760])
    ap.add_argument('--output',type=Path,default=Path('research_output/planning_geometry_m1'))
    ap.add_argument('--base-proxy-radius',type=float,default=None,
                    help='Optional user-validated base footprint radius in metres (same for both robots); not a certified collision model')
    ap.add_argument('--proxy-margin',type=float,default=0.1)
    args=ap.parse_args()
    if args.base_proxy_radius is not None and args.base_proxy_radius<=0: ap.error('proxy radius must be positive')
    if args.proxy_margin<0: ap.error('margin must be nonnegative')
    snapshots=[]
    with h5py.File(args.hdf5,'r') as f:
        obs=f[f'data/{args.demo}/obs']
        n=obs['simulation_time_s'].shape[0]
        for frame in args.frames:
            if frame<0 or frame>=n: ap.error(f'frame {frame} out of bounds 0..{n-1}')
            row=check_snapshot(obs['base_pose_world'][frame],obs['grasp_pose_base'][frame],
                               obs['fingertip_position_base'][frame],
                               obs['fingertip_origin_distance'][frame],
                               obs['gripper_close_command'][frame])
            row.update(frame=int(frame),simulation_time_s=float(obs['simulation_time_s'][frame,0]))
            if args.base_proxy_radius is not None:
                row.update(interpret_proxy(row['base_xy_separation_m'],
                                           args.base_proxy_radius,args.base_proxy_radius,args.proxy_margin))
            else:
                row.update(base_proxy_clearance_m=None,base_proxy_warning=None)
            snapshots.append(row)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'snapshots.json').write_text(json.dumps(snapshots,indent=2,allow_nan=False),encoding='utf8')
    with (args.output/'summary.csv').open('w',newline='',encoding='utf8') as fp:
        cols=['frame','simulation_time_s','geometry_valid','base_xy_separation_m',
              'grasp_origin_separation_world_m','robot0_close','robot1_close',
              'robot0_fingertip_distance_m','robot1_fingertip_distance_m',
              'base_proxy_clearance_m','base_proxy_warning']
        writer=csv.DictWriter(fp,fieldnames=cols);writer.writeheader()
        for s in snapshots:
            writer.writerow(dict(frame=s['frame'],simulation_time_s=s['simulation_time_s'],geometry_valid=s['geometry_valid'],
                base_xy_separation_m=s['base_xy_separation_m'],
                grasp_origin_separation_world_m=s['grasp_origin_separation_world_m'],
                robot0_close=s['robots'][0]['close_intent'],robot1_close=s['robots'][1]['close_intent'],
                robot0_fingertip_distance_m=s['robots'][0]['fingertip_distance_m'],
                robot1_fingertip_distance_m=s['robots'][1]['fingertip_distance_m'],
                base_proxy_clearance_m=s['base_proxy_clearance_m'],base_proxy_warning=s['base_proxy_warning']))
    contract=dict(schema_version='geometry-m1.0',status='MEASUREMENT_ONLY_NOT_PLANNER_READY',
        observations=['base_pose_world (2,7) world XYZ + wxyz','grasp_pose_base (2,7) base XYZ + wxyz',
                      'fingertip_position_base (2,2,3) base meters','joint_position (2,13)',
                      'gripper_close_command (2,1) close intent only'],
        start_reference_frame=int(args.frames[0]),
        grasp_policy='BOTH_CLOSE_INTENT_THROUGHOUT_REORIENTATION',
        target_configuration=None,target_definition_status='UNSPECIFIED',
        geometry_model_status='NOT_VALIDATED',
        collision_model_status='NOT_VALIDATED',
        collision_proxy=('two_planar_circles_visual_DIAGNOSTIC_ONLY' if args.base_proxy_radius else None),
        unknowns=['target joint/base configurations for both robots',
                  'robot collision shape and FK against official articulated asset',
                  'manikin/scene obstacle perception and uncertainty',
                  'cloth constraint feasible set from experiments',
                  'collision checking of whole robot and swept trajectory'],
        dressing_handoff=dict(stage='DRESSING_READY_ONLY_AFTER_VALIDATED_EXECUTION',
                              preserve_gripper_close_intent=True,
                              required=['target_configuration_verified','robot_state','garment_observation',
                                        'collision_clearance_verified','gripper_release_authorized=False']))
    (args.output/'planning_contract_m1.json').write_text(json.dumps(contract,indent=2),encoding='utf8')
    for s in snapshots:
        print(f"frame={s['frame']:4d} t={s['simulation_time_s']:.3f}s valid={s['geometry_valid']} "
              f"base_xy_sep={s['base_xy_separation_m']:.3f}m "
              f"grasp_sep={s['grasp_origin_separation_world_m']}m "
              f"closed={[r['close_intent'] for r in s['robots']]} "
              f"proxy_warning={s['base_proxy_warning']}")
    print('Saved:',args.output.resolve())
    print('PLANNER NOT READY: require validated geometry, collision model and explicit goal.')

if __name__=='__main__':main()
