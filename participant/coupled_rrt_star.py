"""Bounded paired-base RRT* in 4D (x0,y0,x1,y1), guided by coupled A*.

This planner validates only pre-inflated 2D base geometry and pair separation.
It does not certify the arms, garment, joint trajectories, yaw or swept 3D volumes.
RRT* rewiring is deliberately bounded for online computational predictability;
no asymptotic optimality is claimed for a finite budget or biased sampler.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class _Node:
    state: np.ndarray
    parent: int
    cost: float
    children: set = field(default_factory=set)


class CoupledRRTStar:
    def __init__(self, costmap, origin, resolution, *, min_sep=.44,
                 max_sep=.78, max_iterations=180, step_m=.24, seed=17):
        self.costmap = np.asarray(costmap, dtype=np.uint8)
        self.origin = np.asarray(origin, float).reshape(2)
        self.res = float(resolution)
        self.min_sep = float(min_sep)
        self.max_sep = float(max_sep)
        self.n_iter = int(max_iterations)
        self.step_m = float(step_m)
        self.rng = np.random.default_rng(seed)
        if self.costmap.ndim != 2 or self.res <= 0 or self.min_sep <= 0 or self.max_sep <= self.min_sep:
            raise ValueError("invalid costmap or separation bounds")
        if self.n_iter < 0 or self.step_m <= 0:
            raise ValueError("invalid planning budget")
        self.free = self.costmap < 255
        self.free_ij = np.argwhere(self.free)
        self.bounds = self.origin + self.res*np.array(self.costmap.shape)

    def state_valid(self, state):
        p = np.asarray(state, float).reshape(2,2)
        if not np.isfinite(p).all():return False
        d = np.linalg.norm(p[1]-p[0])
        if d < self.min_sep-1e-9 or d > self.max_sep+1e-9:return False
        ij = np.floor((p-self.origin)/self.res).astype(int)
        if np.any(ij < 0) or np.any(ij >= np.array(self.free.shape)):return False
        return bool(self.free[tuple(ij.T)].all())

    def edge_valid(self, a, b):
        """Conservative 2D point-sweep at <= half a map cell, with separation bound."""
        a=np.asarray(a,float).reshape(2,2); b=np.asarray(b,float).reshape(2,2)
        motion=np.linalg.norm(b-a,axis=1)
        n=max(1,int(math.ceil(float(motion.max())/(.5*self.res))))
        if not self.state_valid(a) or not self.state_valid(b):return False
        # Exact minimum center separation for linear synchronized base edges.
        relative_start=a[1]-a[0]
        relative_delta=(b[1]-b[0])-relative_start
        denom=float(relative_delta@relative_delta)
        tau=float(np.clip(-(relative_start@relative_delta)/denom,0,1)) if denom>1e-12 else 0.
        dmin=np.linalg.norm(relative_start+tau*relative_delta)
        if dmin < self.min_sep-1e-9:return False
        if max(np.linalg.norm(relative_start),np.linalg.norm(b[1]-b[0]))>self.max_sep+1e-9:return False
        for k in range(1,n+1):
            mid=a+(b-a)*(k/n)
            if not self.state_valid(mid):return False
        return True

    @staticmethod
    def _dist(a,b):
        return float(np.linalg.norm(np.asarray(a).reshape(2,2)-np.asarray(b).reshape(2,2),axis=1).max())

    def _shortcut(self, path):
        out=[path[0]]; i=0
        while i<len(path)-1:
            for j in range(len(path)-1,i,-1):
                if self.edge_valid(path[i],path[j]):break
            out.append(path[j]);i=j
        return np.asarray(out,float)

    def _cost(self,path):
        return float(sum(self._dist(a,b) for a,b in zip(path,path[1:])))

    def _extract(self,nodes,index):
        path=[]
        while index>=0:
            path.append(nodes[index].state)
            index=nodes[index].parent
        return np.asarray(path[::-1])

    @staticmethod
    def _reparent(nodes,idx,new_parent,new_cost):
        old=nodes[idx].parent
        if old>=0:nodes[old].children.discard(idx)
        nodes[idx].parent=new_parent;nodes[new_parent].children.add(idx)
        delta=new_cost-nodes[idx].cost
        todo=[idx]
        while todo:
            j=todo.pop();nodes[j].cost+=delta
            todo.extend(nodes[j].children)

    @staticmethod
    def _is_ancestor(nodes,ancestor,descendant):
        while descendant>=0:
            if descendant==ancestor:return True
            descendant=nodes[descendant].parent
        return False

    def plan(self,start_xy,goal_xy,*,astar_waypoints=None):
        start=np.asarray(start_xy,float).reshape(2,2)
        goal=np.asarray(goal_xy,float).reshape(2,2)
        if not self.state_valid(start) or not self.state_valid(goal):
            return dict(status="UNKNOWN",reason="start_or_goal_not_known_free")
        hints=None; incumbent=None
        if astar_waypoints is not None:
            hints=np.asarray(astar_waypoints,float)
            if hints.ndim!=3 or hints.shape[1:]!=(2,2):raise ValueError("A* hint shape must be [N,2,2]")
            # Reuse *actual* start and goal, not snapped A* cell centers.
            hint=np.concatenate((start[None],hints,goal[None]),axis=0)
            if all(self.edge_valid(a,b) for a,b in zip(hint,hint[1:])):
                incumbent=self._shortcut(hint)
        if incumbent is None and self.edge_valid(start,goal):incumbent=np.stack((start,goal))
        if not len(self.free_ij):return dict(status="UNKNOWN",reason="no_known_free_cells")
        nodes=[_Node(start.reshape(4),-1,0.)]
        goal_idx=None
        for it in range(self.n_iter):
            u=self.rng.random()
            if u<.18: sample=goal.reshape(4)
            elif u<.77 and hints is not None and len(hints):
                sample=(hints[self.rng.integers(len(hints))]+self.rng.normal(0,.12,(2,2))).reshape(4)
            else:
                ij=self.free_ij[self.rng.integers(len(self.free_ij),size=2)]
                sample=(self.origin+(ij+.5)*self.res).reshape(4)
            distances=np.array([self._dist(node.state,sample) for node in nodes])
            parent=int(distances.argmin())
            ds=sample-nodes[parent].state
            length=self._dist(sample,nodes[parent].state)
            if length>self.step_m:ds*=self.step_m/length
            nxt=nodes[parent].state+ds
            if not self.state_valid(nxt) or not self.edge_valid(nodes[parent].state,nxt):continue
            r=min(.52,max(.16,1.2*(math.log(len(nodes)+2)/(len(nodes)+2))**.25))
            nearby=[j for j,node in enumerate(nodes) if self._dist(node.state,nxt)<=r]
            best_parent=parent;best_cost=nodes[parent].cost+self._dist(nodes[parent].state,nxt)
            for j in nearby:
                c=nodes[j].cost+self._dist(nodes[j].state,nxt)
                if c+1e-8<best_cost and self.edge_valid(nodes[j].state,nxt):
                    best_parent=j;best_cost=c
            idx=len(nodes);nodes.append(_Node(nxt,best_parent,best_cost))
            nodes[best_parent].children.add(idx)
            for j in nearby:
                if j==best_parent or self._is_ancestor(nodes,j,idx):continue
                c=best_cost+self._dist(nxt,nodes[j].state)
                if c+1e-8<nodes[j].cost and self.edge_valid(nxt,nodes[j].state):
                    self._reparent(nodes,j,idx,c)
            if self._dist(nxt,goal)<=self.step_m and self.edge_valid(nxt,goal):
                goal_candidate=best_cost+self._dist(nxt,goal)
                if goal_idx is None:
                    goal_idx=len(nodes)
                    nodes.append(_Node(goal.reshape(4),idx,goal_candidate))
                    nodes[idx].children.add(goal_idx)
                elif goal_candidate+1e-8<nodes[goal_idx].cost:
                    self._reparent(nodes,goal_idx,idx,goal_candidate)
        best=incumbent
        if goal_idx is not None:
            tree=self._shortcut(self._extract(nodes,goal_idx).reshape(-1,2,2))
            if best is None or self._cost(tree)+1e-8<self._cost(best):best=tree
        if best is None:
            return dict(status="UNKNOWN",reason="rrt_star_budget_or_no_base_path",iterations=self.n_iter,
                        nodes=len(nodes),planner="COUPLED_RRT_STAR_4D")
        # Caller must never use this as a 3D/cloth safety authorization.
        return dict(status="BASE_PATH_CANDIDATE",waypoints=best.tolist(),
                    length_cost_m=self._cost(best),iterations=self.n_iter,nodes=len(nodes),
                    improved_over_astar=bool(incumbent is not None and self._cost(best)<self._cost(incumbent)-1e-5),
                    planner="COUPLED_RRT_STAR_4D",safety_scope="2D_BASE_ONLY",
                    arm_collision="UNKNOWN",garment_safety="UNKNOWN")
