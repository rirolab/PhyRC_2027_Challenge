"""Cold-start Strategy Decision Layer (SDL) for a *pair* of mobile bases.

Map -> candidate generation -> coupled A* -> bounded coupled RRT*.
Experience is an optional ranking hint; it never authorizes a motion.
Outputs remain BASE_PATH_CANDIDATE unless a separate full-body validator proves safe.
"""
from __future__ import annotations
import math
import numpy as np
from strategy_core import CoupledBaseAStar, canonical_scene, select_strategy_priors
from coupled_rrt_star import CoupledRRTStar
from conditional_prior import tactical_candidates


def _rotate(v, a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[c,-s],[s,c]])@v


class StrategyDecisionLayer:
    def __init__(self, *, a_star_max_expansions=2200, rrt_iterations=120,
                 min_base_sep=.43, span_extension_m=.16, max_trials=7):
        self.astar_budget=int(a_star_max_expansions)
        self.rrt_budget=int(rrt_iterations)
        self.min_base_sep=float(min_base_sep)
        self.span_extension_m=float(span_extension_m)
        self.max_trials=int(max_trials)

    def candidates(self, base_xy, *, goal_xy=None, gaps=None, experience_matches=None, conditional_intent=None):
        start=np.asarray(base_xy,float).reshape(2,2)
        center=start.mean(axis=0)
        offset=(start[1]-start[0])/2
        # Swap is a *hypothesized subgoal*, never a verified dressing endpoint.
        swap=start[::-1] if goal_xy is None else np.asarray(goal_xy,float).reshape(2,2)
        result=tactical_candidates(start,conditional_intent or {})
        result.append(dict(family="COUPLED_SWAP_DIRECT",phase_goals=[swap],
                     target_basis="relative_pair_exchange_hypothesis" if goal_xy is None else "provided_goal"))
        # Convert at most two relative keypoint hints to *proposed* via-goals;
        # transfer requires fresh map-based A* validation. Never execute or
        # directly replay recorded absolute poses.
        for match in (experience_matches or [])[:2]:
            if (match.get('symmetry_transform')!='identity' or
                match.get('geometry_certification')!='PASSAGE_2D_REVIEWED_NO_FULL_BODY_CERT'):
                continue
            prior=match.get('relative_waypoint_prior') or {}
            points=np.asarray(prior.get('base_relative_waypoints',[]),float)
            if points.ndim!=3 or points.shape[1:]!=(2,2) or len(points)<3 or not np.isfinite(points).all():
                continue
            original=match.get('reference_pair_distance_m')
            if not original or not (.8<=float(np.linalg.norm(start[1]-start[0]))/float(original)<=1.2):
                continue
            scale=float(np.linalg.norm(start[1]-start[0]))/float(original)
            theta=float(np.arctan2((start[1]-start[0])[1],(start[1]-start[0])[0]))
            c,s=math.cos(theta),math.sin(theta)
            rot=np.array([[c,-s],[s,c]])
            idx=np.unique(np.rint(np.linspace(1,len(points)-2,min(2,len(points)-2))).astype(int))
            transferred=[center + scale*(points[j]@rot.T) for j in idx]
            # These hints are optional intermediate goals; all are rechecked.
            result.append(dict(family='EXPERIENCE_WAYPOINT_GUIDE',
                passer=match.get('passer'),yielding_robot=match.get('yielding_robot'),
                phase_goals=[*transferred,swap],
                target_basis='conditional_relative_prior_requires_new_scene_geometry'))
        for side,ang in (("CCW",math.pi/2),("CW",-math.pi/2)):
            half=_rotate(offset,ang)
            mid=np.vstack((center-half,center+half))
            result.append(dict(family="COUPLED_ROTATION_"+side,phase_goals=[mid,swap],
                               target_basis="relative_pair_exchange_hypothesis"))
        # A short local yield does not imply a verified passage by a table.
        unit=(start[1]-start[0])/max(1e-9,2*np.linalg.norm(offset))
        for yielding in (0,1):
            y=start.copy();y[yielding]+=(-unit if yielding==0 else unit)*.07
            result.append(dict(family="YIELD_PASS_CANDIDATE",passer=1-yielding,
                               yielding_robot=yielding,phase_goals=[y,swap],
                               target_basis="hypothesized_yield_no_table_identity"))
        return result

    def decide(self, costmap, origin, resolution, base_xy, *,
               experience_library=None, gaps=None, goal_xy=None, seed=17, rejected_keys=None, conditional_intent=None):
        cost=np.asarray(costmap,dtype=np.uint8)
        xy=np.asarray(base_xy,float).reshape(2,2)
        distance=float(np.linalg.norm(xy[1]-xy[0]))
        if cost.ndim!=2 or not np.isfinite(xy).all() or not np.isfinite(distance):
            return dict(status="INVALID_INPUT",candidates=[],best=None)
        known_cells=int((cost<255).sum())
        if known_cells==0:return dict(status="ACTIVE_OBSERVATION_REQUIRED",reason="no_free_map_cells",candidates=[],best=None)
        if distance<self.min_base_sep:
            return dict(status="HOLD_REPLAN",reason="pair_base_separation_under_research_threshold",candidates=[],best=None)
        if not bool((cost<255).any()):
            return dict(status="ACTIVE_OBSERVATION_REQUIRED",candidates=[],best=None)
        width=[g.get("observed_width_m") for g in (gaps or []) if g.get("base_passable")=="TRUE" and g.get("observed_width_m") is not None]
        scene=canonical_scene(xy,min(width) if width else None)
        priors=select_strategy_priors(scene,experience_library or [])
        # 2D conservative separation bound only; does not certify grasp geometry.
        min_sep=self.min_base_sep
        max_sep=max(distance+.06, distance+self.span_extension_m)
        tested=[]
        # Keep both YIELD/PASS role hypotheses eligible even when two
        # experience waypoint proposals are present. The old first-five
        # truncation silently dropped either or both yielding alternatives.
        proposals=self.candidates(xy,goal_xy=goal_xy,gaps=gaps,
                                   experience_matches=priors,conditional_intent=conditional_intent)[:self.max_trials]
        for idx,p in enumerate(proposals):
            key=p['family']+':p'+str(p.get('passer'))
            if key in (rejected_keys or set()):
                tested.append(dict(family=p['family'],passer=p.get('passer'),status='REJECTED_EPISODIC',reason='same_scene_prior_failed'))
                continue
            current=xy.copy();whole=[current.copy()];expanded=0;ok=True;reason=""
            prior=next((v for v in priors if v.get("passer")==p.get("passer") and p.get("passer") is not None),None)
            for stage in p["phase_goals"]:
                planner=CoupledBaseAStar(cost,origin,resolution,step_cells=1,
                            min_base_separation=min_sep,max_base_separation=max_sep,
                            max_expansions=self.astar_budget)
                a=planner.plan(current,stage,prior=prior)
                expanded+=int(a.get("expanded",0))
                if a.get("status")!="BASE_PATH_CANDIDATE":
                    ok=False;reason=a.get("reason","a_star_failed");break
                # Require the real initial and target states, not only A* cell centers.
                w=np.asarray(a["waypoints"],float)
                whole.extend(w);whole.append(np.asarray(stage,float))
                current=np.asarray(stage,float)
            if not ok:
                tested.append(dict(family=p["family"],passer=p.get("passer"),yielding_robot=p.get("yielding_robot"),
                                   status="UNKNOWN",reason=reason,astar_expansions=expanded))
                continue
            warm=np.asarray(whole,float)
            rrt=CoupledRRTStar(cost,origin,resolution,min_sep=min_sep,max_sep=max_sep,
                  max_iterations=self.rrt_budget,seed=seed+idx)
            rr=rrt.plan(xy,current,astar_waypoints=warm)
            if rr.get("status")!="BASE_PATH_CANDIDATE":
                tested.append(dict(family=p["family"],status="UNKNOWN",reason=rr.get("reason"),astar_expansions=expanded))
                continue
            # RRT* can shortcut a valid A* path but it is not allowed to erase
            # tactical via-goals if they were essential to the chosen tactic.
            route=np.asarray(rr["waypoints"],float)
            if len(p["phase_goals"])>1:
                # Conservative tactic preservation: use stage-segmented A* route.
                # Report as A* warm-start, not as RRT* optimized route.
                if all(rrt.edge_valid(a,b) for a,b in zip(warm,warm[1:])):
                    route=warm
                    rr["planner"]="COUPLED_ASTAR_PHASED_RRTSTAR_DIAGNOSTIC"
                else:
                    tested.append(dict(family=p["family"],status="UNKNOWN",reason="invalid_via_goal_edge"));continue
            length=float(np.linalg.norm(np.diff(route,axis=0),axis=2).max(axis=1).sum()) if len(route)>1 else 0.
            same_role=next((v for v in priors if v.get("passer")==p.get("passer") and p.get("passer") is not None),None)
            # Experience tiny tie-breaker only; never influences validity.
            score=length+.0004*expanded-.02*(float(same_role["priority"]) if same_role else 0.)
            tested.append(dict(family=p["family"],passer=p.get("passer"),yielding_robot=p.get("yielding_robot"),
                               status="BASE_PATH_CANDIDATE",planner=rr["planner"],cost=float(score),
                               path_length_m=length,astar_expansions=expanded,
                               rrt_iterations=rr.get("iterations"),waypoints=route.tolist(),
                               phase_goals=[np.asarray(g).tolist() for g in p["phase_goals"]],
                               safety_scope="BASE_2D_ONLY",arm_ik="UNKNOWN",garment="UNKNOWN",
                               experience_support=bool(same_role),target_basis=p["target_basis"],
                               strategy_id=p.get("strategy_id"),
                               role_basis=p.get("role_basis","UNCONDITIONED_GEOMETRIC_HYPOTHESIS")))
        feasible=[x for x in tested if x["status"]=="BASE_PATH_CANDIDATE"]
        # A conditional role is a tactical preference, not permission for fallback
        # to be mislabeled as the same tactic. Prefer an actually feasible
        # role-consistent candidate; keep geometry-only fallback explicitly tagged.
        selected_role=(conditional_intent or {}).get('yielding_robot')
        preferred=[v for v in feasible if selected_role is not None and
                   v.get('yielding_robot')==selected_role]
        best=min(preferred,key=lambda c:c['cost']) if preferred else (
            min(feasible,key=lambda c:c['cost']) if feasible else None)
        return dict(status="BASE_PATH_CANDIDATE" if best else "ACTIVE_OBSERVATION_OR_REPLAN",
                    conditional_intent=conditional_intent,
                    conditional_role_plan_found=bool(preferred),
                    role_plan_status=("ROLE_CONSISTENT_2D_CANDIDATE" if preferred else
                        "ALTERNATIVE_2D_ONLY_NO_ROLE_MATCH" if feasible and selected_role is not None else
                        "UNCONDITIONED_SCENE_UNKNOWN" if selected_role is None else "NO_FEASIBLE_ROLE_PLAN"),
                    reason=None if best else "no_certified_known_free_paired_base_path",
                    best=best,candidates=tested,known_cells=known_cells,
                    experience_candidates=len(priors),scene=scene,
                    no_experience_required=True,
                    full_robot_execution_authorized=False)
