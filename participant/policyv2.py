"""V1-grasp + episode-local mapping + SDL + coupled A*/RRT* (PhyRC policy).

Default PLAN_ONLY: V1 autonomously grasps; mapping starts at frame 0; once grasp
is acknowledged by V1, an SDL base-path is logged without commanding base
motion. SIM experimental base mode is opt-in, NOT full-body certified.
"""
from __future__ import annotations
import json
import os
import hashlib
import uuid
from pathlib import Path
import numpy as np
import original_v1
from mapper import Mapper, Geometry, ORIGIN, RES, SHAPE
from costmap_memory import detect_gaps, frontier_targets
from strategy_decision import StrategyDecisionLayer
from strategy_core import load_library
from strategy_memory import DecisionMemory
from trajectory_executor import PairedBaseExecutor
from conditional_prior import choose_intent, choose_proximity_intent
from scene_proximity import GarmentProximityEstimator
from traversability_evidence import build_grounded_evidence, start_causal_audit, fuse_traversal_with_rgbd
from video_capture import VideoCapture

ROOT=Path(__file__).resolve().parent


class PolicyV2:
    def __init__(self,v1,output,*,map_stride=12,plan_stride=100,
                 strategy_library=None,execute_mode='plan_only',
                 rrt_iterations=100,astar_expansions=1700,
                 diagnostic_scene_side=None,diagnostic_ack=False,
                 video_enabled=True,video_stride=4):
        self.v1=v1;self.root=Path(output)
        self.map_stride=max(1,int(map_stride));self.plan_stride=max(10,int(plan_stride))
        self.library=load_library(strategy_library)
        self.geometry=Geometry()  # robot meshes for self-depth filtering; not privileged object meshes
        self.sdl=StrategyDecisionLayer(rrt_iterations=int(rrt_iterations),
                                       a_star_max_expansions=int(astar_expansions))
        self.executor=PairedBaseExecutor(mode=execute_mode)
        self.execute_mode=execute_mode
        if diagnostic_scene_side is not None and (execute_mode!='plan_only' or not diagnostic_ack):
            raise ValueError('diagnostic scene hint requires plan_only and explicit diagnostic_ack')
        if diagnostic_scene_side not in (None,'LEFT','RIGHT'):
            raise ValueError('diagnostic_scene_side must be null, LEFT or RIGHT')
        self.diagnostic_scene_side=diagnostic_scene_side
        self.video_enabled=bool(video_enabled);self.video_stride=max(1,int(video_stride))
        self.reset()

    def reset(self):
        self.v1.reset();self.map=None;self.memory=None;self.video=None
        self.frame=0;self.holding=False;self.handoff_info=None
        self.last_plan_frame=-10**9;self.last_plan=None
        self.scene_signature=None;self.selected_candidate_key=None
        self.map_error_count=0;self.planning_error_count=0
        self.executor.reset()
        self.proximity=GarmentProximityEstimator()
        self.intent=choose_intent(self.diagnostic_scene_side,
            origin='DIAGNOSTIC_MANUAL_SCENE_HINT_NOT_INFERRED' if self.diagnostic_scene_side else 'NO_VALIDATED_HEAD_WRIST_SCENE_CLASSIFIER')

    def _new_run(self):
        self.root.mkdir(parents=True,exist_ok=True)
        path=self.root/uuid.uuid4().hex
        self.map=Mapper(path,self.geometry,original_v1.masks,'actor_public',review_regions=None)
        self.memory=DecisionMemory(path)
        if self.video_enabled:self.video=VideoCapture(path,stride=self.video_stride)
        self.memory.record(dict(frame=0,event='EPISODE_START',mode=self.execute_mode,
                                mapping_from_frame=0,experience_entries=len(self.library),
                                conditional_intent=self.intent,video_enabled=self.video_enabled))

    def _save_event(self,event,**kw):
        if self.memory is not None:self.memory.record(dict(frame=self.frame,event=event,**kw))

    def _map(self,obs,*,force=False):
        if not force and self.frame%self.map_stride!=0:return
        if self.map.last==self.frame:return
        try:
            self.map.observe(obs,self.frame)
            # Observe the garment only before *either* robot has begun closing
            # its gripper. After contact the cloth moves with the pair and
            # proximity is not a reliable scene-initialization condition.
            arms=getattr(self.v1,'arms',[])
            precontact=not any(getattr(a,'s',None) in ('CLOSE','WAIT','HOLD','CHECK','DRESS') for a in arms)
            if (self.execute_mode=='plan_only' and self.diagnostic_scene_side is None
                    and not self.holding and precontact):
                evidence=self.proximity.update(
                    frame=self.frame,
                    head_evidence=getattr(self.map,'last_garment_evidence',[]),
                    base_pose_world=obs['base_pose_world'])
                if evidence is not None:
                    self.intent=choose_proximity_intent(evidence)
                    self._save_event('SCENE_ROLE_AUTO_SELECTED',evidence=evidence,intent=self.intent)
            # Large compressed voxel arrays are deliberately NOT written every step.
            if self.frame==0 or self.frame%600==0:self.map.snapshot('periodic_'+str(self.frame))
        except Exception as e:
            self.map_error_count+=1
            self._save_event('MAPPING_ERROR',reason=repr(e),count=self.map_error_count)

    def _handoff(self,obs):
        # V1 only switches to DRESS after BOTH arms were HOLD for HOLD_S.
        arms=getattr(self.v1,'arms',[])
        hold_times=[getattr(a,'hold_t',None) for a in arms]
        first_ok=[getattr(a,'first_ok',None) for a in arms]
        t=float(np.asarray(obs['simulation_time_s']).reshape(-1)[0])
        evidence=dict(v1_mode='DRESS',dual_holding_earlier=len(arms)==2 and all(v is not None for v in hold_times),
                      v1_first_visual_checks=first_ok,hold_start_times=hold_times,
                      physical_grasp_verified=False,level='V1_VISUAL_HOLD_AND_WAIT_NOT_GROUND_TRUTH')
        self.handoff_info=dict(frame=self.frame,time_s=t,grasp_evidence=evidence,
                    base_pose_world=np.asarray(obs['base_pose_world'],float).reshape(2,7).tolist(),
                    joint_position=np.asarray(obs['joint_position'],float).reshape(2,13).tolist())
        self._map(obs,force=True)
        try:
            self.map.handoff(obs,self.frame,'v1_dual_hold_visual_evidence')
        except Exception as e:
            self._save_event('HANDOFF_SNAPSHOT_ERROR',reason=repr(e))
        self.holding=True
        self._save_event('V1_V2_HANDOFF',evidence=evidence,conditional_intent=self.intent,
                         proximity_estimator=self.proximity.latest)

    def _plan(self,obs):
        self.last_plan_frame=self.frame
        self._save_event('SDL_ROLE_INTENT',intent=self.intent,
            warning=('user-provided diagnostic hint is NOT scene perception' if self.diagnostic_scene_side else
                     None if self.intent.get('role_selected') else 'role not inferred'))
        if self.map is None or self.map.last<0:
            self._save_event('PLAN_BLOCKED',reason='no_valid_mapping_yet');return
        try:
            proj=self.map.current_base_map()
            costs=np.asarray(proj['base_cost_2d'],np.uint8)
            known=np.asarray(proj['base_known_free_2d'],bool)
            occupied=np.asarray(proj['base_occupied_2d'],bool)
            gaps=detect_gaps(known,occupied,ORIGIN[:2],RES,max_candidates=6)
            current=np.asarray(obs['base_pose_world'],float).reshape(2,7)[:,:2]
            self.scene_signature=hashlib.blake2b(costs.tobytes()+
                np.floor(current/.10).astype(np.int16).tobytes(),digest_size=10).hexdigest()
            excluded=self.memory.rejected_for_scene(self.scene_signature)
            decision=self.sdl.decide(costs,ORIGIN[:2],RES,current,
                experience_library=self.library,gaps=gaps,seed=self.frame+17,
                rejected_keys=excluded,conditional_intent=self.intent)
            self.last_plan=decision
            best=decision.get('best')
            # Persist precisely the currently-known causal 2D map and the
            # planned pair waypoints for visual offline inspection. These are
            # proposed routes, not V2 execution or full-body certifications.
            plans_dir=self.map.out/'plans'
            plans_dir.mkdir(exist_ok=True)
            stem=f'plan_{self.frame:06d}'
            np.savez_compressed(plans_dir/(stem+'_map.npz'),
                base_cost_2d=costs,base_unknown_2d=proj['base_unknown_2d'],
                base_occupied_2d=occupied,origin=ORIGIN[:2],resolution=RES,
                source_frame=int(self.map.last))
            # Review layer is intentionally separated from the strict map.
            # Even if a hypothetical route exists, it is never loaded into
            # PairedBaseExecutor; no automatic motion is authorized.
            if self.execute_mode=='plan_only':
                try:
                    review=self.map.current_review_map()
                    np.savez_compressed(plans_dir/(stem+'_review_map.npz'),
                        review_cost_2d=review['base_review_cost_2d'],
                        strict_cost_2d=review['base_strict_cost_2d'],
                        hypothesis_only_2d=review['base_review_hypothesis_only_2d'],
                        occupied_2d=review['base_review_obstacle_2d'],
                        observed_free_2d=review['base_review_observed_free_2d'],
                        floor_votes_2d=review['base_review_floor_votes_2d'],
                        origin=ORIGIN[:2],resolution=RES,
                        source_frame=int(self.map.last))
                    xy_ij=np.floor((current-ORIGIN[:2])/RES).astype(int)
                    start_info=[]
                    for idx in range(2):
                        ij=xy_ij[idx]
                        inside=bool(np.all(ij>=0) and np.all(ij<costs.shape))
                        start_info.append(dict(robot=idx,xy=current[idx].tolist(),
                            inside_map=inside,
                            strict_free=bool(costs[tuple(ij)]<255) if inside else False,
                            review_candidate=bool(review['base_review_cost_2d'][tuple(ij)]<255) if inside else False,
                            unverified=bool(review['base_review_hypothesis_only_2d'][tuple(ij)]) if inside else True))
                    # Evidence-gated map: requires measured/interpolated floor
                    # AND at least one observed free ray in the base height band.
                    # This is still NOT a safety-certified path. Never pass it to
                    # SDL execution or PairedBaseExecutor.
                    grounded=build_grounded_evidence(proj,review,
                        base_visits=self.map.base_visit_count_2d)
                    np.savez_compressed(plans_dir/(stem+'_grounded_map.npz'),
                        grounded_cost_2d=grounded['grounded_cost_2d'],
                        evidence_tier_2d=grounded['grounded_evidence_tier_2d'],
                        measured_floor_2d=grounded['grounded_measured_floor_2d'],
                        interpolated_floor_2d=grounded['grounded_interpolated_floor_2d'],
                        ray_free_2d=grounded['grounded_ray_free_2d'],
                        base_visit_count_2d=grounded['grounded_base_visit_count_2d'],
                        origin=ORIGIN[:2],resolution=RES,
                        source_frame=int(self.map.last))
                    # V4 diagnostic-only traversal/RGB-D fusion; distinct
                    # from the strict map passed to SDL and PairedBaseExecutor.
                    fused=fuse_traversal_with_rgbd(proj,review,grounded,
                        centerline_count_by_robot=self.map.traversal.centerline_count_by_robot,
                        footprint_count_by_robot=self.map.traversal.footprint_count_by_robot)
                    np.savez_compressed(plans_dir/(stem+'_traversal_fusion_map.npz'),
                        strict_cost_2d=costs,
                        fused_review_cost_2d=fused['traversal_fused_cost_2d'],
                        traversed_centerline_2d=fused['traversal_observed_centerline_2d'],
                        traversed_footprint_2d=fused['traversal_observed_footprint_2d'],
                        conflict_2d=fused['traversal_blocked_by_rgbd_2d'],
                        origin=ORIGIN[:2],resolution=RES,
                        source_frame=int(self.map.last),
                        pose_observations=int(len(self.map.traversal.frames)))
                    self._save_event('TRAVERSAL_FUSION_REVIEW',
                        scope='PLAN_ONLY_NOT_COLLISION_CERTIFIED',
                        pose_observations=int(len(self.map.traversal.frames)),
                        observed_centerline_cells=int(fused['traversal_observed_centerline_2d'].sum()),
                        observed_footprint_cells=int(fused['traversal_observed_footprint_2d'].sum()),
                        rgbd_conflict_cells=int(fused['traversal_blocked_by_rgbd_2d'].sum()),
                        fused_candidates=int((fused['traversal_fused_cost_2d']<255).sum()),
                        fused_unverified=int(fused['traversal_unverified_2d'].sum()),
                        discontinuities=self.map.traversal.discontinuities,
                        fused_map=str((plans_dir/(stem+'_traversal_fusion_map.npz')).relative_to(self.map.out)))
                    z=self.map.full_3d.spec.origin[2]+(
                        np.arange(self.map.full_3d.spec.shape[2])+.5)*self.map.full_3d.spec.resolution
                    height_band=(z>=.10)&(z<=.65)
                    hv=self.map.full_3d.layers()['unknown'][:,:,height_band].sum(axis=2)
                    height_unknown=np.repeat(np.repeat(hv,2,axis=0),2,axis=1)
                    floor_cube=self.map.full.floor.reshape(SHAPE[0],2,SHAPE[1],2)
                    start_audit=start_causal_audit(current,ORIGIN[:2],RES,proj,review,grounded,
                        strict_floor_min=floor_cube.min(axis=(1,3)),
                        height_unknown_count=height_unknown)
                    self._save_event('RGBD_GROUNDED_EVIDENCE',
                        mode='PLAN_ONLY_NOT_MOTION_CERTIFIED',
                        grounded_candidates=int((grounded['grounded_cost_2d']<255).sum()),
                        grounded_unverified=int(grounded['grounded_unverified_2d'].sum()),
                        measured_floor=int(grounded['grounded_measured_floor_2d'].sum()),
                        interpolated_floor=int(grounded['grounded_interpolated_floor_2d'].sum()),
                        free_ray_cells=int(grounded['grounded_ray_free_2d'].sum()),
                        starts=start_audit,
                        grounded_map=str((plans_dir/(stem+'_grounded_map.npz')).relative_to(self.map.out)))
                    self._save_event('TRAVERSABILITY_REVIEW',
                        scope='PLAN_ONLY_UNVERIFIED_NOT_FOR_EXECUTION',
                        strict_traversable_cells=int((costs<255).sum()),
                        review_candidate_cells=int((review['base_review_cost_2d']<255).sum()),
                        hypothesis_only_cells=int(review['base_review_hypothesis_only_2d'].sum()),
                        observed_floor_cells=int((review['base_review_floor_votes_2d']>0).sum()),
                        start=start_info,
                        review_map=str((plans_dir/(stem+'_review_map.npz')).relative_to(self.map.out)))
                except Exception as exc:
                    self._save_event('TRAVERSABILITY_REVIEW_ERROR',reason=repr(exc))
            plan_summary=dict(frame=int(self.frame),map_frame=int(self.map.last),
                plan_status=decision['status'],scene=decision.get('scene'),
                known_cells=int((costs<255).sum()),
                unknown_cells=int(proj['base_unknown_2d'].sum()),
                strategy=best['family'] if best else None,
                candidate_details=[dict(family=c['family'],status=c['status'],
                    reason=c.get('reason')) for c in decision.get('candidates',[])],
                best_waypoints=best['waypoints'] if best else [],
                conditional_intent=self.intent,role_plan_status=decision.get('role_plan_status'),
                selected_role_matches_prior=(bool(best and best.get('yielding_robot')==self.intent.get('yielding_robot'))
                    if self.intent.get('role_selected') else None),
                phase_goals=best.get('phase_goals',[]) if best else [],
                full_robot_execution_authorized=False)
            path=plans_dir/(stem+'.json')
            temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(plan_summary,ensure_ascii=False,
                allow_nan=False,indent=2),encoding='utf-8')
            os.replace(temporary,path)
            self._save_event('SDL_DECISION',status=decision['status'],
                known_cells=plan_summary['known_cells'],unknown_cells=plan_summary['unknown_cells'],
                gap_count=len(gaps),experience_candidates=decision.get('experience_candidates',0),
                strategy=plan_summary['strategy'],conditional_intent=self.intent,
                role_plan_status=decision.get('role_plan_status'),
                base_waypoint_count=len(best['waypoints']) if best else 0,
                plan_file=str(path.relative_to(self.map.out)),
                candidates=plan_summary['candidate_details'])
            if best is not None:
                self.selected_candidate_key=best['family']+':p'+str(best.get('passer'))
                if not self.executor.set_plan(obs,best['waypoints']):
                    self._save_event('EXECUTOR_PLAN_REJECTED')
            else:
                # never replay an obsolete path when current map invalidates it
                self.executor.path=None
                self.selected_candidate_key=None
                targets=frontier_targets(known,proj['base_unknown_2d'],ORIGIN[:2],RES,current[0],max_results=4)
                self._save_event('ACTIVE_OBSERVATION_REQUEST',frontiers=targets,
                    comment='viewpoints_only_no_uncertified_auto_exploration')
        except Exception as e:
            self.planning_error_count+=1
            self.executor.path=None;self.last_plan=None
            self.selected_candidate_key=None
            self._save_event('PLANNER_ERROR',reason=repr(e),count=self.planning_error_count)

    def __call__(self,obs):
        if self.map is None:self._new_run()
        # Full policy-rate pose history, including SCAN/NAV and blind camera areas.
        # This records public observations only; never changes motion commands.
        try:
            self.map.record_pose(obs,self.frame)
        except Exception as exc:
            self._save_event('TRAVERSAL_MEMORY_ERROR',reason=repr(exc))
        self._map(obs)
        if not self.holding:
            previous=getattr(self.v1,'mode',None)
            action=self.v1(obs)  # V1 autonomously handles SCAN and GRASP unchanged.
            if previous=='GRASP' and getattr(self.v1,'mode',None)=='DRESS':
                self._handoff(obs)
                # Do not let V1's fixed dressing KF move after transition.
                action=self.executor.hold()
        else:
            if self.frame-self.last_plan_frame>=self.plan_stride:
                self._plan(obs)
            action=self.executor.hold()
            if self.executor.path is not None and self.map.last>=0:
                try:
                    cost=self.map.current_base_map()['base_cost_2d']
                    action=self.executor.step(obs,cost,ORIGIN[:2],RES)
                    if self.executor.last_status.startswith('FAIL_STOP'):
                        reason=self.executor.last_status
                        if self.scene_signature and self.selected_candidate_key:
                            self.memory.failure(self.frame,self.scene_signature,self.selected_candidate_key,reason)
                        self._save_event('EXECUTOR_STATUS',status=reason)
                        self.executor.path=None;self.selected_candidate_key=None
                    elif self.frame%40==0:
                        self._save_event('EXECUTOR_STATUS',status=self.executor.last_status)
                except Exception as e:
                    action=self.executor.hold()
                    self._save_event('EXECUTOR_ERROR',reason=repr(e))
        # Video is sampled from the same current public observation passed to V1/SDL.
        # Video failure is observable, but must not cause non-neutral V2 actions.
        if self.video is not None:
            try:
                self.video.push(obs,self.frame,mode=getattr(self.v1,'mode','UNKNOWN'),
                    holding=self.holding,intent=self.intent,decision=self.last_plan)
            except Exception as e:
                self._save_event('VIDEO_CAPTURE_ERROR',reason=repr(e))
        self.frame+=1
        return np.asarray(action,np.float32).reshape(2,9)


def load_policy(checkpoint,observation_space,action_space):
    # Official run.sh forwards STRETCH4_* but not arbitrary V2_* host variables.
    # Read an explicit, versionable configuration mounted as /project:ro.
    settings_path=ROOT/'v2_settings.json'
    cfg=json.loads(settings_path.read_text()) if settings_path.exists() else {}
    ck=Path(checkpoint) if checkpoint else ROOT/'checkpoint.json'
    v1=original_v1.load_policy(str(ck),observation_space,action_space)
    mode=os.getenv('V2_EXECUTION_MODE',cfg.get('execution_mode','plan_only'))
    ack=os.getenv('V2_SIM_EXPERIMENT_ACK','1' if cfg.get('sim_experiment_ack') else '0')
    if mode=='sim_base_experimental' and ack!='1':
        raise ValueError('Experimental base control requires sim_experiment_ack=true')
    if mode not in ('plan_only','sim_base_experimental'):
        raise ValueError('execution_mode must be plan_only or sim_base_experimental')
    return PolicyV2(v1,os.getenv('V2_MAP_OUT',cfg.get('map_out','/output/v2_sdl')),
                    map_stride=int(os.getenv('V2_MAP_STRIDE',cfg.get('map_stride',12))),
                    plan_stride=int(os.getenv('V2_PLAN_STRIDE',cfg.get('plan_stride',120))),
                    strategy_library=os.getenv('V2_STRATEGY_LIBRARY',cfg.get('strategy_library')),
                    execute_mode=mode,
                    rrt_iterations=int(os.getenv('V2_RRT_ITERATIONS',cfg.get('rrt_iterations',60))),
                    astar_expansions=int(os.getenv('V2_ASTAR_EXPANSIONS',cfg.get('astar_expansions',650))),
                    diagnostic_scene_side=cfg.get('diagnostic_scene_side'),
                    diagnostic_ack=bool(cfg.get('diagnostic_ack',False)),
                    video_enabled=bool(cfg.get('video_enabled',True)),
                    video_stride=int(cfg.get('video_stride',4)))
