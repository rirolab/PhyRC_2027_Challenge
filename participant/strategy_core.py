"""Pure V2 strategy primitives: event mining, scene-conditioned priors, base planning.

NO privileged simulation state. The pair planner checks *only* mapped 2D base
space and a conservative proxy for inter-base separation. It does NOT certify
IK, arms, grasp orientations, cloth dynamics or the full collision envelope.
"""
from __future__ import annotations
import heapq
import json
import math
from pathlib import Path
import numpy as np


def _runs(mask, min_frames=3):
    x=np.asarray(mask, bool).reshape(-1)
    d=np.diff(np.r_[False,x,False].astype(np.int8))
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)) if b-a>=min_frames]


def segment_motion(t, xy, yaw, grasp_span, closed=None, *, speed=.055,
                   turn_rate=.16, min_duration_s=.2, close_stability_frames=3):
    """Multi-label motion hypotheses relative to the *candidate* grasp frame.

    `closed` is command intent, NOT observed successful cloth grasp.
    Before grasp, mapping may proceed but exchange events must not be mined.
    Overlapping YIELD/PASS/TURN windows are allowed; semantic gap proof is
    provided separately by the causal RGB-D/map evidence validator.
    """
    t=np.asarray(t,float).reshape(-1)
    xy=np.asarray(xy,float)
    yaw=np.asarray(yaw,float)
    sep=np.asarray(grasp_span,float).reshape(-1)
    if (xy.shape!=(len(t),2,2) or yaw.shape!=(len(t),2) or sep.shape!=(len(t),)
            or len(t)<2 or np.any(np.diff(t)<=0) or not np.isfinite(xy).all()
            or not np.isfinite(yaw).all() or not np.isfinite(sep).all()):
        raise ValueError('invalid full-rate motion series')
    anchor=None;anchor_evidence='UNKNOWN_NO_CLOSE_CHANNEL'
    if closed is not None:
        c=np.asarray(closed).astype(bool)
        if c.shape!=(len(t),2):
            raise ValueError('closed must be [frames,2], command intent only')
        both=c.all(axis=1)
        length=max(1,int(close_stability_frames))
        stable=np.convolve(both.astype(int),np.ones(length,int),'valid')
        starts=np.flatnonzero(stable==length)
        if len(starts):
            anchor=int(starts[0]);anchor_evidence='DUAL_CLOSE_COMMAND_ONLY'
        else:
            anchor_evidence='NO_SUSTAINED_DUAL_CLOSE'
    dt=np.diff(t)
    dx=np.diff(xy,axis=0)
    vel=dx/dt[:,None,None]
    speeds=np.linalg.norm(vel,axis=2)
    yr=np.diff(yaw,axis=0)/dt[:,None]
    base_rel=xy[:,1]-xy[:,0]
    pair_angle=np.unwrap(np.arctan2(base_rel[:,1],base_rel[:,0]))
    pair_rate=np.diff(pair_angle)/dt
    overlap=(speeds[:,0]>speed)&(speeds[:,1]>speed)
    after=np.arange(len(dt))>=anchor if anchor is not None else np.zeros(len(dt),bool)
    events=[]
    for passer in (0,1):
        y=1-passer
        heading=np.stack((np.cos(yaw[:-1,y]),np.sin(yaw[:-1,y])),1)
        retreat=-np.sum(vel[:,y]*heading,axis=1)
        yield_mask=after & (retreat>speed) & (speeds[:,passer]<max(speed*2,.12))
        # Partner need NOT remain still: a PASS movement may overlap YIELD/TURN.
        pass_mask=after & (speeds[:,passer]>speed) & ((speeds[:,passer]>.75*speeds[:,y]) | (retreat>.3*speed))
        for label,mask in (('YIELD_CANDIDATE',yield_mask),('PASS_CANDIDATE',pass_mask)):
            for a,b in _runs(mask,2):
                if t[b]-t[a]>=min_duration_s:
                    events.append(dict(label=label,start_frame=a,end_frame=b,
                         duration_s=float(t[b]-t[a]),passer=passer,yielding_robot=y,
                         displacement_m=float(np.linalg.norm(dx[a:b,passer],axis=1).sum()),
                         status='KINEMATIC_HYPOTHESIS',gap_identity='UNKNOWN'))
    # A relative orbital turn can overlap both yield and pass; also track yaw.
    turn_mask=after & ((np.abs(pair_rate)>turn_rate)&(overlap | (np.abs(yr).max(axis=1)>turn_rate)))
    for a,b in _runs(turn_mask,2):
        if t[b]-t[a]>=min_duration_s:
            events.append(dict(label='COUPLED_TURN',start_frame=a,end_frame=b,
                 duration_s=float(t[b]-t[a]),turn_deg=float(np.rad2deg(pair_angle[b]-pair_angle[a])),
                 yaw_change_deg=np.rad2deg(yaw[b]-yaw[a]).tolist(),
                 span_change_m=float(np.max(sep[a:b+1])-np.min(sep[a:b+1])),
                 status='KINEMATIC_HYPOTHESIS'))
    jointly_moving=after & overlap & (np.abs(pair_rate)<turn_rate*.5)
    for a,b in _runs(jointly_moving,2):
        if t[b]-t[a]>=min_duration_s:
            events.append(dict(label='JOINT_TRANSLATION',start_frame=a,end_frame=b,
                 duration_s=float(t[b]-t[a]),status='KINEMATIC_HYPOTHESIS'))
    if anchor is not None:
        initial=base_rel[anchor]
        denom=max(float(initial@initial),1e-9)
        for f in range(anchor+1,len(t)):
            if float(base_rel[f]@initial/denom)<-.5:
                events.append(dict(label='RELATIVE_EXCHANGE',start_frame=anchor,end_frame=f,
                                    duration_s=float(t[f]-t[anchor]),
                                    reference_frame=anchor,
                                    status='PAIR_REVERSAL_NOT_TASK_COMPLETION'))
                break
    events.sort(key=lambda v:(v['start_frame'],v['end_frame'],v['label']))
    ref=anchor if anchor is not None else 0
    return dict(events=events,grasp_reference=dict(frame=anchor,
                 time_s=None if anchor is None else float(t[anchor]),evidence=anchor_evidence,
                 physical_grasp_verified=False),
                metrics=dict(path_m=np.linalg.norm(dx[ref:],axis=2).sum(axis=0).tolist(),
                 maximum_grasp_span_deviation_m=float(np.max(abs(sep[ref:]-sep[ref]))),
                 coordination_overlap=float(np.mean(overlap[ref:])) if len(overlap[ref:]) else 0.,
                 pair_rotation_deg=float(np.rad2deg(pair_angle[-1]-pair_angle[ref])),
                 duration_s=float(t[-1]-t[ref])),
                method='full_rate_multilabel_from_grasp_candidate_not_semantic_certification')


def canonical_scene(base_xy, gap_width_m=None, *, garment_side=None):
    base=np.asarray(base_xy,float)
    if base.shape!=(2,2) or not np.isfinite(base).all():
        raise ValueError("two 2D base poses required")
    return dict(pair_distance_m=float(np.linalg.norm(base[1]-base[0])),
                gap_width_m=float(gap_width_m) if gap_width_m is not None else None,
                garment_side=garment_side or "UNKNOWN")


def select_strategy_priors(scene, library, *, max_results=5):
    """Similarity may influence priority, but cannot certify any candidate."""
    ranked=[]
    for p in library:
        if p.get("split") not in ("train", "validated_training"):
            continue
        if p.get("qualification")!="OFFICIAL_SUCCESS":
            continue
        conditions=p.get("conditions",{})
        err=0.
        matched=0
        for key,scale in (("pair_distance_m",.5),("gap_width_m",.4)):
            v=scene.get(key);ref=conditions.get(key)
            if v is not None and ref is not None:
                err+=min(5.,abs(float(v)-float(ref))/scale)
                matched+=1
        a,b=scene.get("garment_side"),conditions.get("garment_side")
        if a not in (None,"UNKNOWN") and b not in (None,"UNKNOWN"):
            err+=(0 if a==b else 3)
            matched+=1
        if not matched:
            continue
        support=max(1,int(p.get("support_count",1)))
        weight=float(p.get("diagnostic_weight",.5))
        base_priority=float(math.exp(-err/max(matched,1))*math.sqrt(support)*max(0.,min(1.,weight)))
        template=dict(strategy_family=p.get("strategy_family","UNKNOWN"),
                      similarity=float(math.exp(-err/max(matched,1))),
                      safety_status="UNVERIFIED_IN_CURRENT_SCENE",
                      geometry_certification=p.get('geometry_certification','UNKNOWN'),
                      relative_waypoint_prior=p.get('relative_waypoint_prior'),
                      reference_pair_distance_m=conditions.get('pair_distance_m'),
                      evidence_source=p.get("provenance",{}))
        ranked.append(dict(template,strategy_id=p.get("strategy_id","unknown"),
                           passer=p.get("passer"),yielding_robot=p.get("yielding_robot"),
                           symmetry_transform="identity",priority=base_priority))
        # Mirroring is only a *hypothesis*: never infer mirrored IK or wrist reachability.
        if p.get("passer") in (0,1) and a in (None,"UNKNOWN"):
            ranked.append(dict(template,strategy_id=p.get("strategy_id","unknown")+":mirrored",
                               passer=1-int(p["passer"]),yielding_robot=int(p["passer"]),
                               symmetry_transform="swap_roles_unverified",priority=.9*base_priority))
    return sorted(ranked,key=lambda p:(-p["priority"],p["strategy_id"]))[:max_results]


class CoupledBaseAStar:
    """Bounded search over (x0,y0,x1,y1); useful without any strategy library.

    `costmap` must be fail-closed and already inflated for a circular base.
    The output is a geometric *proposal*, not authorized actuator commands.
    """
    def __init__(self,costmap,origin_xy,resolution,*,step_cells=2,
                 min_base_separation=.50,max_base_separation=None,max_expansions=6000):
        costmap=np.asarray(costmap)
        if costmap.ndim!=2 or resolution<=0 or step_cells<1:
            raise ValueError("invalid costmap")
        self.raw=costmap
        self.step=int(step_cells)
        # Requiring EVERY subcell to be known keeps downsampling conservative.
        nx,ny=costmap.shape[0]//self.step,costmap.shape[1]//self.step
        block=costmap[:nx*self.step,:ny*self.step].reshape(nx,self.step,ny,self.step)
        self.free=(block<255).all(axis=(1,3))
        self.penalty=block.astype(float).mean(axis=(1,3))/253
        self.origin=np.asarray(origin_xy,float)
        self.res=float(resolution)*self.step
        self.sep_min=float(min_base_separation)
        self.sep_max=max_base_separation
        self.max_expansions=int(max_expansions)
        self.dirs=((0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1))

    def cell(self, xy):
        return tuple(np.floor((np.asarray(xy)-self.origin)/self.res).astype(int))

    def world(self, cell):
        return (self.origin + (np.asarray(cell,dtype=float)+.5)*self.res).tolist()

    def _valid(self,pair):
        a,b=pair[:2],pair[2:]
        for p in (a,b):
            if min(p)<0 or p[0]>=self.free.shape[0] or p[1]>=self.free.shape[1] or not self.free[p]:
                return False
        d=np.linalg.norm((np.asarray(a)-np.asarray(b))*self.res)
        return d >= self.sep_min and (self.sep_max is None or d<=self.sep_max)

    def _edge(self,state,nxt):
        # Sample the simultaneous center-to-center sweep at least each quarter cell.
        p=np.asarray(state).reshape(2,2).astype(float)
        q=np.asarray(nxt).reshape(2,2).astype(float)
        for robot in range(2):
            da=(q[robot]-p[robot]).astype(int)
            if abs(da[0])==1 and abs(da[1])==1:
                # No cutting through diagonal corners adjacent to unknown/occupied.
                x,y=p[robot].astype(int)
                if not (self._valid_one((x+int(da[0]),y)) and
                        self._valid_one((x,y+int(da[1])))):
                    return False
        # Check exact minimum simultaneous inter-base separation over the
        # complete linear edge; four discrete samples can miss a close pass.
        d0 = (p[1] - p[0]) * self.res
        dd = ((q[1] - q[0]) - (p[1] - p[0])) * self.res
        norm2 = float(dd @ dd)
        u_min = float(np.clip(-float(d0 @ dd) / norm2, 0., 1.)) if norm2 > 1e-12 else 0.
        d_min = float(np.linalg.norm(d0 + u_min * dd))
        if d_min < self.sep_min - 1e-9:
            return False
        if self.sep_max is not None and max(float(np.linalg.norm(d0)), float(np.linalg.norm(d0+dd))) > self.sep_max + 1e-9:
            return False
        for u in (.25,.5,.75,1.):
            interpolated=self.origin + ((1-u)*p+u*q+.5)*self.res
            if np.linalg.norm(interpolated[0]-interpolated[1]) < self.sep_min:
                return False
            cells=np.floor((interpolated-self.origin)/self.res).astype(int)
            if any(not self._valid_one(tuple(c)) for c in cells):
                return False
        return True

    def _valid_one(self,p):
        return 0<=p[0]<self.free.shape[0] and 0<=p[1]<self.free.shape[1] and bool(self.free[p])

    def plan(self,start_xy,goal_xy,*,prior=None):
        start=np.asarray(start_xy,float)
        goal=np.asarray(goal_xy,float)
        if start.shape!=(2,2) or goal.shape!=(2,2):
            raise ValueError("start/goal must be [2,2]")
        s=(*self.cell(start[0]),*self.cell(start[1]))
        g=(*self.cell(goal[0]),*self.cell(goal[1]))
        if not self._valid(s) or not self._valid(g):
            return dict(status="UNKNOWN",reason="start_or_goal_not_known_free",expanded=0)
        def heuristic(st):
            p=np.asarray(st).reshape(2,2);q=np.asarray(g).reshape(2,2)
            return float(np.linalg.norm(p-q,axis=1).max())
        counter=0
        queue=[(heuristic(s),counter,s)]
        best={s:0.};parent={s:None}
        solved=None;expanded=0
        while queue and expanded<self.max_expansions:
            f,_,cur=heapq.heappop(queue)
            if f > best[cur]+heuristic(cur)+1e-6:continue
            if cur==g:
                solved=cur;break
            expanded+=1
            for da in self.dirs:
                for db in self.dirs:
                    if da==(0,0) and db==(0,0):continue
                    nxt=(cur[0]+da[0],cur[1]+da[1],cur[2]+db[0],cur[3]+db[1])
                    if not self._valid(nxt) or not self._edge(cur,nxt):continue
                    move=max(math.hypot(*da),math.hypot(*db))
                    new_g=best[cur]+move+.025*(self.penalty[nxt[:2]]+self.penalty[nxt[2:]])
                    if new_g>=best.get(nxt,float('inf')):continue
                    best[nxt]=new_g;parent[nxt]=cur
                    # Priors affect *ordering*, never bypass the feasibility predicate.
                    bias=0.
                    if prior and prior.get("passer") in (0,1):
                        passer=int(prior["passer"])
                        moved_a=da!=(0,0);moved_b=db!=(0,0)
                        if (moved_a if passer==0 else moved_b):bias=-.015
                    counter+=1
                    heapq.heappush(queue,(new_g+heuristic(nxt)+bias,counter,nxt))
        if solved is None:
            return dict(status="UNKNOWN",reason="budget_or_no_base_path",expanded=expanded)
        states=[]
        while solved is not None:
            states.append(solved);solved=parent[solved]
        states.reverse()
        poses=np.asarray([[self.world(p[:2]),self.world(p[2:])] for p in states])
        return dict(status="BASE_PATH_CANDIDATE",waypoints=poses.tolist(),
                    expanded=expanded,used_experience_as_ordering_only=bool(prior),
                    clearance_certificate="NOT_FULL_ROBOT_VALIDATION",
                    arm_status="UNKNOWN",garment_status="UNKNOWN")


def load_library(path):
    if path is None:
        return []
    path=Path(path)
    if not path.is_file():
        return []
    data=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data,list):
        raise ValueError("strategy library must be a JSON list")
    return data


def safe_mark_object_cells(mask, xy, origin_xy, resolution):
    """Rasterize visible world XY points without wrapping or out-of-map indexing."""
    cells=np.floor((np.asarray(xy,float).reshape(-1,2)-np.asarray(origin_xy))/resolution+1e-10).astype(np.int64)
    inside=((cells>=0)&(cells<np.asarray(mask.shape))).all(axis=1)
    if inside.any():
        mask[tuple(cells[inside].T)]=True
    return int(inside.sum())
