"""Bounded base-pair tracking adapter for Isaac Sim research runs.

EXPERIMENTAL simulation-only mode. It checks 2D map cells, center separation,
observed gripper span and prospective local gripper span. This is NOT whole-body
IK/arm/garment/continuous-collision validation. The default mode never moves.
"""
from __future__ import annotations
import math
import numpy as np


def yaw(q):
    w,x,y,z=np.asarray(q,float).reshape(4)
    return float(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def wrap(a):return float((a+math.pi)%(2*math.pi)-math.pi)


def grasp_points(obs):
    """Public grasp_pose_base to approximated 3D world grasp point."""
    bp=np.asarray(obs['base_pose_world'],float).reshape(2,7)
    gp=np.asarray(obs['grasp_pose_base'],float).reshape(2,7)
    out=[]
    for i in (0,1):
        # Full quaternion multiplication using scipy; permitted proprioception.
        from scipy.spatial.transform import Rotation
        r=Rotation.from_quat(bp[i,[4,5,6,3]])
        out.append(bp[i,:3]+r.apply(gp[i,:3]))
    return np.asarray(out)


class PairedBaseExecutor:
    def __init__(self, *, mode='plan_only', speed_limit=.10,
                 grasp_span_tolerance_m=.07, min_sep=.43, waypoint_tol=.075):
        if mode not in ('plan_only','sim_base_experimental'):
            raise ValueError('invalid execution mode')
        self.mode=mode
        self.limit=float(speed_limit)
        self.span_tol=float(grasp_span_tolerance_m)
        self.min_sep=float(min_sep)
        self.tol=float(waypoint_tol)
        self.reset()

    def reset(self):
        self.path=None;self.index=0;self.initial_grasp_span=None
        self.reference_yaw=None;self.reference_joints=None
        self.last_status='UNINITIALIZED'

    @staticmethod
    def hold():
        a=np.zeros((2,9),np.float32)
        a[:,8]=0.  # official contract: 0 keeps the previous gripper intent.
        return a

    def set_plan(self,obs,waypoints):
        path=np.asarray(waypoints,float)
        if path.ndim!=3 or path.shape[1:]!=(2,2) or len(path)<2 or not np.isfinite(path).all():
            return False
        self.path=path;self.index=1
        bp=np.asarray(obs['base_pose_world'],float).reshape(2,7)
        self.reference_yaw=np.array([yaw(v[3:7]) for v in bp])
        self.reference_joints=np.asarray(obs['joint_position'],float).reshape(2,13).copy()
        try:
            g=grasp_points(obs)
            self.initial_grasp_span=float(np.linalg.norm(g[1]-g[0]))
        except (KeyError,ValueError):
            self.initial_grasp_span=None
        self.last_status='PATH_LOADED'
        return True

    @staticmethod
    def _known(cost,origin,res,poses):
        ij=np.floor((np.asarray(poses)-np.asarray(origin))/res).astype(int)
        return bool(np.all(ij>=0) and np.all(ij<np.asarray(cost.shape)) and
                    np.all(cost[tuple(ij.T)]<255))

    def step(self,obs,costmap,origin,res):
        cmd=self.hold()
        if self.mode!='sim_base_experimental':
            self.last_status='PLAN_ONLY';return cmd
        if self.path is None or self.reference_joints is None:
            self.last_status='NO_PLAN';return cmd
        bp=np.asarray(obs['base_pose_world'],float).reshape(2,7)
        joints=np.asarray(obs['joint_position'],float).reshape(2,13)
        xy=bp[:,:2]
        yaws=np.array([yaw(v[3:7]) for v in bp])
        if (not np.isfinite(bp).all() or not np.isfinite(joints).all() or
            np.linalg.norm(xy[0]-xy[1])<self.min_sep or
            np.any(np.abs([wrap(a-b) for a,b in zip(yaws,self.reference_yaw)])>.27)):
            self.last_status='FAIL_STOP_POSE';return cmd
        if not self._known(costmap,origin,res,xy):
            self.last_status='FAIL_STOP_MAP_UNKNOWN';return cmd
        try:
            g=grasp_points(obs)
            d=float(np.linalg.norm(g[1]-g[0]))
            if self.initial_grasp_span is None or abs(d-self.initial_grasp_span)>self.span_tol:
                self.last_status='FAIL_STOP_GRASP_SPAN';return cmd
        except (KeyError,ValueError):
            self.last_status='FAIL_STOP_NO_GRASP_POSE';return cmd
        while self.index<len(self.path) and np.max(np.linalg.norm(self.path[self.index]-xy,axis=1))<self.tol:
            self.index+=1
        if self.index>=len(self.path):
            self.last_status='BASE_ROUTE_COMPLETE';return cmd
        tgt=self.path[self.index]
        dist=tgt-xy
        v=np.clip(.8*dist,-self.limit,self.limit)
        # Prospective estimated grasp positions at next 0.05 s; a proxy only.
        predict=g.copy();predict[:,:2]+=v*.05
        if (abs(np.linalg.norm(predict[1]-predict[0])-self.initial_grasp_span)>self.span_tol or
                np.linalg.norm((xy+v*.05)[1]-(xy+v*.05)[0])<self.min_sep):
            self.last_status='FAIL_STOP_PREDICTED_SPAN';return cmd
        # The route is center-only. Prevent movement into unknown at short lookahead.
        if not self._known(costmap,origin,res,xy+v*.3):
            self.last_status='FAIL_STOP_AHEAD_UNKNOWN';return cmd
        for r in (0,1):
            cs,sn=math.cos(yaws[r]),math.sin(yaws[r])
            cmd[r,0]=np.clip(v[r,0]*cs+v[r,1]*sn,-self.limit,self.limit)
            cmd[r,1]=np.clip(-v[r,0]*sn+v[r,1]*cs,-self.limit,self.limit)
            cmd[r,2]=np.clip(1.1*wrap(self.reference_yaw[r]-yaws[r]),-.12,.12)
            # Hold current V1 lift/arm/wrist targets. Normalization is V1-style
            # and must be checked against actual controller response on VM.
            lift_err=self.reference_joints[r,3]-joints[r,3]
            arm_err=(self.reference_joints[r,4:8]-joints[r,4:8]).sum()
            cmd[r,3]=np.clip(3.*lift_err/1.54,-.14,.14)
            cmd[r,4]=np.clip(3.*arm_err/1.21,-.14,.14)
            cmd[r,5:8]=np.clip(1.2*(self.reference_joints[r,8:11]-joints[r,8:11]),-.12,.12)
        self.last_status='SIM_EXPERIMENTAL_TRACKING'
        return cmd
