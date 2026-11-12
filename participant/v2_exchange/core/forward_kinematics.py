"""Read-only offline Forward Kinematics for PhyRC Stretch 4 USD articulation.

Uses only authored USD Physics joint frames and joint positions. No Isaac Sim needed.
No collision validity or cloth grasping claim is made by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


def qrot(wxyz):
    q = np.asarray(wxyz, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError('Invalid quaternion')
    w,x,y,z = q/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def mat(pos=(0,0,0), quat=(1,0,0,0)):
    t = np.eye(4)
    t[:3,:3] = qrot(quat)
    t[:3,3] = np.asarray(pos, dtype=float).reshape(3)
    return t


def inverse(t):
    out=np.eye(4)
    out[:3,:3]=t[:3,:3].T
    out[:3,3]=-out[:3,:3] @ t[:3,3]
    return out


def motion(kind, axis, q):
    a={'X': np.array([1.,0,0]), 'Y':np.array([0.,1,0]), 'Z':np.array([0.,0,1])}[axis.upper()]
    out=np.eye(4)
    if kind=='prismatic':
        out[:3,3]=a*q
    elif kind=='revolute':
        c,s=np.cos(q),np.sin(q)
        ax=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
        out[:3,:3]=np.eye(3)*c+(1-c)*np.outer(a,a)+s*ax
    elif kind!='fixed':
        raise ValueError(f'Unsupported joint kind {kind}')
    return out


@dataclass(frozen=True)
class Joint:
    name:str
    parent:str
    child:str
    kind:str
    axis:str
    t0:np.ndarray
    t1:np.ndarray
    lower:float=float('-inf')
    upper:float=float('inf')


def fwd_edge(j:Joint, value:float):
    return j.t0 @ motion(j.kind,j.axis,value) @ inverse(j.t1)


def forward_graph(joints, root, q_by_name):
    """Return transforms relative to root, using parent->child joint edges.
    Raises if duplicate child or unreachable requested grasp link is later found.
    """
    children=defaultdict(list)
    parents={}
    for j in joints:
        if j.child in parents:
            raise ValueError(f'Joint child has multiple parents: {j.child}: {parents[j.child]} & {j.name}')
        parents[j.child]=j.name
        children[j.parent].append(j)
    poses={root:np.eye(4)}
    pending=[root]
    while pending:
        p=pending.pop(0)
        for j in children[p]:
            q=0. if j.kind=='fixed' else q_by_name[j.name]
            if j.child in poses:
                raise ValueError(f'Cycle in articulation: {j.child}')
            poses[j.child]=poses[p] @ fwd_edge(j,q)
            pending.append(j.child)
    return poses


def read_usd_joints(usd_path):
    from pxr import Usd,UsdPhysics
    stage=Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f'Cannot load USD {usd_path}')
    result=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint=UsdPhysics.Joint(prim)
        b0=list(joint.GetBody0Rel().GetTargets())
        b1=list(joint.GetBody1Rel().GetTargets())
        # A joint anchored to the world, rather than between two bodies,
        # is not part of the link-to-link traversal.
        if len(b0)!=1 or len(b1)!=1:
            continue
        name=prim.GetName()
        if prim.IsA(UsdPhysics.PrismaticJoint):
            kind='prismatic'; typed=UsdPhysics.PrismaticJoint(prim)
        elif prim.IsA(UsdPhysics.RevoluteJoint):
            kind='revolute'; typed=UsdPhysics.RevoluteJoint(prim)
        elif prim.IsA(UsdPhysics.FixedJoint):
            kind='fixed'; typed=None
        else:
            raise ValueError(f'Unsupported USD joint {name}: {prim.GetTypeName()}')
        axis=str(typed.GetAxisAttr().Get()) if typed else 'X'
        # pxr.Gf.Quatf GetReal/GetImaginary -> WXYZ
        def pose(pos,quat):
            p=np.asarray(pos,dtype=float)
            return mat(p,[float(quat.GetReal()),*list(quat.GetImaginary())])
        t0=pose(joint.GetLocalPos0Attr().Get(),joint.GetLocalRot0Attr().Get())
        t1=pose(joint.GetLocalPos1Attr().Get(),joint.GetLocalRot1Attr().Get())
        lower,upper=float('-inf'),float('inf')
        if typed:
            lo=typed.GetLowerLimitAttr().Get()
            hi=typed.GetUpperLimitAttr().Get()
            if lo is not None: lower=float(lo)
            if hi is not None: upper=float(hi)
            if kind=='revolute':
                lower,upper=np.deg2rad([lower,upper])
        result.append(Joint(name,str(b0[0]),str(b1[0]),kind,axis,t0,t1,lower,upper))
    return result


def angle_error_deg(A,B):
    r=A[:3,:3].T@B[:3,:3]
    return float(np.rad2deg(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))


def pose_error(pred, expected):
    return float(np.linalg.norm(pred[:3,3]-expected[:3,3])),angle_error_deg(pred,expected)
