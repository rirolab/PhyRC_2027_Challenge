"""Shared, read-only Isaac measurements and recording for teleop/policies."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from .evaluation import Phase1Scorer, ScoringConfig, format_score_items, RATE_RULE
from .evaluation_geometry import GarmentGeometry, vneck_front_vertices
from .state import array, rotation
from .evaluation_clock import ContactClock
from .evaluation_contact import CollisionContact, contact_context_from_stage


def grasp_health(cloths, rigs, stage):
    """Measure real FEM attachment-anchor error, not just the close command."""
    points = array(cloths[0].get_world_positions())[0]
    result = []
    for rig in rigs:
        state = rig['state']; grab = state.get('grabbed')
        local = state.get('_native_local_offsets'); mask = state.get('_grab_anchor_mask')
        root = rig['robot'].prim_path.rsplit('/', 1)[0]
        if grab is None or local is None or mask is None or not stage.GetPrimAtPath(root + '/ClothGraspAttachment'):
            result.append({'attached': False, 'anchor_p95_error_m': None}); continue
        tf = array(rig['robot']._articulation_view._physics_view.get_link_transforms())[0, rig['grasp_link_idx']]
        R = rotation(np.r_[tf[6], tf[3:6]])
        mask = array(mask).astype(bool); ids = array(grab[1]).astype(int)[mask]
        expected = tf[:3] + np.asarray(local)[mask] @ R.T
        error = np.linalg.norm(points[ids] - expected, axis=1)
        result.append({'attached': True, 'anchor_count': len(ids), 'anchor_p95_error_m': float(np.quantile(error, .95))})
    return result


class IsaacMeasurements:
    """Current single-shirt, statically posed mannequin measurement backend."""
    def __init__(self, backend, cloths, rigs, module):
        from pxr import Usd, UsdGeom, UsdSkel
        self.backend, self.cloths, self.rigs, self.M = backend, cloths, rigs, module
        if len(cloths) != 1:
            raise ValueError('Phase 1 measurement currently supports one garment per episode')
        self.stage = backend.stage
        mesh = UsdGeom.Mesh(cloths[0].prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError('Expected triangle garment mesh')
        rest = np.asarray(mesh.GetPointsAttr().Get(), float)
        self.geometry = GarmentGeometry(rest, np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3),
                                        front_ids=vneck_front_vertices(mesh))
        self.pickup_references = {}
        skeleton = next((UsdSkel.Skeleton(p) for p in Usd.PrimRange(self.stage.GetPrimAtPath('/World/Human'))
                         if p.IsA(UsdSkel.Skeleton)), None)
        if skeleton is None:
            raise ValueError('Human skeleton missing')
        posed = skeleton.GetPrim().GetAttribute('phyrc:posedJointPositions').Get()
        if posed is None:
            raise ValueError('Posed evaluation landmarks missing; restart using updated teleop source')
        names = [str(j).rsplit('/', 1)[-1] for j in skeleton.GetJointsAttr().Get()]
        matrix = np.asarray(UsdGeom.Xformable(skeleton).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        joints = np.c_[np.asarray(posed), np.ones(len(posed))] @ matrix
        joints = dict(zip(names, joints[:, :3]))
        self.neck_chain = np.array([joints[name] for name in ('spine3', 'neck', 'head')])
        human_mesh, local, _, _, hands, _ = module._hand_vertex_sets(self.stage, '/World/Human', .948514)
        root = np.asarray(UsdGeom.Xformable(self.stage.GetPrimAtPath('/World/Human'))
                          .ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        human = (np.c_[local, np.ones(len(local))] @ root)[:, :3]
        mesh_joints = UsdSkel.BindingAPI(human_mesh).GetJointsAttr().Get() or skeleton.GetJointsAttr().Get()
        joint_names = [str(j).rsplit('/', 1)[-1] for j in mesh_joints]
        indices = np.asarray(human_mesh.GetAttribute('primvars:skel:jointIndices').Get()).reshape(len(human), -1)
        weights = np.asarray(human_mesh.GetAttribute('primvars:skel:jointWeights').Get()).reshape(indices.shape)
        dominant = indices[np.arange(len(human)), weights.argmax(1)]
        head_ids = [i for i, name in enumerate(joint_names) if name in ('head', 'jaw', 'left_eye', 'right_eye')]
        self.head_surface = human[np.isin(dominant, head_ids)]
        if not len(self.head_surface):
            raise ValueError('Cannot identify head surface for collar completion')
        self.arms = {}
        for side in ('left', 'right'):
            shoulder, elbow, wrist = [joints[f'{side}_{joint}'] for joint in ('shoulder', 'elbow', 'wrist')]
            hand = human[hands[side]]
            axis = wrist - elbow
            axis /= np.linalg.norm(axis)
            distances = (hand - wrist) @ axis
            tip = hand[distances >= np.quantile(distances, .9)].mean(0)
            self.arms[side] = np.array([shoulder, elbow, wrist, tip])
        self.table_centres = np.asarray(backend.garment_table_centers)
        self.table_size = np.asarray(module.BOX_SIZE)
        self.clearance_m = 0.01
        self.max_anchor_error_m = 0.035
        self.pickup_rise_m = 0.05
        self.pickup_diagnostics = []
        self.contact_context = contact_context_from_stage(self.stage, str(cloths[0].prim.GetPath()))
        self.contact_detector = CollisionContact(self.contact_context, root)
        self.metadata = {
            'measurement_version': RATE_RULE,
            'neck_clearance_normal_basis': 'anatomical_chest_to_head',
            'coverage_definition': 'binary upper-arm enclosure by its routed sleeve; hand exits its cuff; torso and forearm excluded',
            'pickup_definition': 'same healthy attachment for 3s with median anchor rise >=5cm from that grasp acquisition; hem may remain on table',
            'pickup_rise_m': self.pickup_rise_m,
            'neck_chain_world_m': self.neck_chain.tolist(),
            'coverage_samples_per_arm': 40,
            'clearance_m': self.clearance_m,
            'max_anchor_p95_error_m': self.max_anchor_error_m,
            'cuff_planarity_limit': 0.6,
            'arms_world_m': {side: chain.tolist() for side, chain in self.arms.items()},
            'garment_topology_sha256': hashlib.sha256(self.geometry.faces.tobytes()).hexdigest(),
            'geometry_source_sha256': hashlib.sha256(Path(__file__).with_name('evaluation_geometry.py').read_bytes()).hexdigest(),
            'live_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scoring_source_sha256': hashlib.sha256(Path(__file__).with_name('evaluation.py').read_bytes()).hexdigest(),
        }
        self.metadata.update(contact_criterion=self.contact_context['criterion'],
                             native_physx_contact_event=False,
                             contact_limitations=self.contact_context['limitations'],
                             contact_source_sha256=hashlib.sha256(Path(__file__).with_name('evaluation_contact.py').read_bytes()).hexdigest())

    def contact(self, points):
        return self.contact_detector.touches(points)

    def physical_sample(self):
        """Lightweight pickup evidence sampled at EVERY physics step."""
        points = array(self.cloths[0].get_world_positions())[0]
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite garment state during evaluation')
        health = grasp_health(self.cloths, self.rigs, self.stage)
        anchors = []
        for rig, h in zip(self.rigs, health):
            state = rig['state']
            ids = (array(state['grabbed'][1]).astype(int)[array(state['_grab_anchor_mask']).astype(bool)]
                   if h['attached'] and h.get('anchor_count', 0) > 0 else np.array([], dtype=int))
            anchors.append((int(state['grabbed'][0]) if len(ids) else -1, ids))
        return self.physical_from_state(points, health, anchors)

    def physical_from_state(self, points, health, anchors):
        """Identical pickup logic for live physics and recorded physics states."""
        holding = [bool(h['attached'] and h.get('anchor_count', 0) > 0
                        and h['anchor_p95_error_m'] <= self.max_anchor_error_m) for h in health]
        lifted = []
        self.pickup_diagnostics = []
        for i, ((cloth_id, ids), h, held) in enumerate(zip(anchors, health, holding)):
            rise = 0.0
            if h['attached'] and h.get('anchor_count', 0) > 0:
                signature = (cloth_id, ids.tobytes())
                if i not in self.pickup_references or self.pickup_references[i][0] != signature:
                    self.pickup_references[i] = (signature, points[ids, 2].copy())
                rise = float(np.median(points[ids, 2] - self.pickup_references[i][1]))
            else:
                self.pickup_references.pop(i, None)
            lifted.append(bool(held and rise >= self.pickup_rise_m))
            self.pickup_diagnostics.append(dict(h, lifted=lifted[-1], anchor_median_rise_m=rise))
        clear = bool(points[:, 2].min() > self.clearance_m)
        # Triangle AABBs conservatively include triangles spanning a tabletop
        # even when none of their vertices projects onto that table.
        triangles = points[self.geometry.faces]
        lower, upper = triangles.min(1), triangles.max(1)
        for centre in self.table_centres:
            lo, hi = centre - self.table_size / 2, centre + self.table_size / 2
            overlap = ((upper[:, :2] >= lo[:2] - self.clearance_m)
                       & (lower[:, :2] <= hi[:2] + self.clearance_m)).all(1)
            if np.any(overlap & (lower[:, 2] <= hi[2] + self.clearance_m)):
                clear = False
        return points, holding, clear, lifted

    def measure(self, physical=None):
        points, holding, clear, lifted = physical if physical is not None else self.physical_sample()
        wrists, coverage, details = self.geometry.measure(points, self.arms, self.neck_chain, self.head_surface)
        left, right = self.arms['left'][0], self.arms['right'][0]
        lateral = left - right
        lateral /= np.linalg.norm(lateral)
        beyond = {'left': bool(np.any((points - left) @ lateral > 0)),
                  'right': bool(np.any((points - right) @ (-lateral) > 0))}
        quality = {}
        if 'orientation' in details:
            quality = {'upper_arm_coverage': {side: details[side]['upper_arm_coverage'] for side in self.arms},
                       'neck_out': details['neck']['neck_out'],
                       'front_facing': details['orientation']['front_facing']}
        if quality and getattr(self.geometry, 'sleeve_regions', None) is not None:
            quality.update(rate_rule=RATE_RULE, neck_passed=details['neck']['neck_out'], upper_arm_sleeve_covered={s: details[s]['upper_arm_sleeve_covered'] for s in self.arms},
                           hand_out_of_sleeve={s: details[s]['hand_out'] for s in self.arms})
        return {'gripper_holding': holding, 'garment_lifted_clear': clear,
                'gripper_lifted': lifted, 'pickup_diagnostics': self.pickup_diagnostics,
                'dressing_complete': details['dressing_complete'],
                'wrist_in_sleeve': wrists, 'garment_beyond_shoulder': beyond,
                'arm_coverage': coverage, 'geometry_diagnostics': details, **quality}


class EvaluationSession:
    """Start/finish one continuous attempt; samples and result are durable files.

    Can be used by either the GUI loop or any DressingEnv policy rollout.
    Geometry is measured at 20 Hz, while pickup interruptions are checked at
    physics frequency and propagated conservatively into the next sample.
    """
    def __init__(self, backend, cloths, rigs, module, output='/output/evaluation'):
        self.backend, self.cloths, self.rigs, self.module = backend, cloths, rigs, module
        self.output = Path(output)
        self.active = False
        self.subscription = None
        self.path = None
        self.last_report = None

    @classmethod
    def for_policy(cls, env, output='/output/evaluation'):
        return cls(env.backend, env.cloths, env.rigs, env.M, output)

    def start(self, metadata=None, *, measurements=None, subscribe=True):
        if self.active:
            raise RuntimeError('An evaluation is already running')
        self.measurements = (measurements if measurements is not None else
                             IsaacMeasurements(self.backend, self.cloths, self.rigs, self.module))
        self.scorer = Phase1Scorer(ScoringConfig(max_sample_gap_s=0.05))
        self.dt = float(self.backend.world.get_physics_dt())
        self.stride = round(0.05 / self.dt)
        if not np.isclose(self.stride * self.dt, 0.05):
            raise ValueError('Evaluation requires a physics rate divisible by 20 Hz')
        self.ticks = 0
        self.clock = ContactClock(self.dt)
        self.interval_holding = [True, True]
        self.interval_clear = True
        self.interval_lifted = [True, True]
        self.trace = []
        self._last_progress_key = None
        physical = self.measurements.physical_sample()
        self.clock.update(0, self.measurements.contact(physical[0]))
        initial = self.measurements.measure(physical)
        self.metadata = dict(metadata or {})
        self.metadata.update(self.measurements.metadata)
        self.metadata['started_with_arm_inserted'] = any(initial['wrist_in_sleeve'].values())
        self.metadata['started_in_contact'] = self.clock.first_contact_tick == 0
        self.path = self.output / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:6])
        self.path.mkdir(parents=True, exist_ok=False)
        self.stream = (self.path / 'samples.jsonl').open('w', encoding='utf-8')
        self.active = True
        print(f'[Evaluation] START {self.path}', flush=True)
        self._record(initial)
        if subscribe:
            self.subscription = self.backend.world._physics_context._physics_sim_interface.subscribe_physics_on_step_events(
                pre_step=False, order=200, on_update=self._physics_step)
        return self.path

    def _record(self, sample):
        sample = dict(sample, time_s=self.ticks * self.dt,
                      first_contact_time_s=self.clock.result(0)['first_contact_time_s'],
                      physics_tick=self.ticks, physics_dt_s=self.dt,
                      first_contact_tick=self.clock.first_contact_tick)
        self.scorer.update(sample)
        self.trace.append(sample)
        self.stream.write(json.dumps(sample, allow_nan=False) + '\n')
        self.stream.flush()
        if self.active:
            self._print_progress()

    def clock_result(self, points):
        result = self.clock.result(points['raw_points'])
        if self.scorer.rate_rule == RATE_RULE:
            result.update(self.scorer.rate_result())
        return result

    def _print_progress(self):
        result = self.scorer.points()
        items = format_score_items(result)
        # Elapsed time, hold counters and rate change continuously, even when
        # nothing new has been achieved. They must not trigger repeated lines.
        key = (items, self.clock.first_contact_tick)
        if key == self._last_progress_key:
            return
        self._last_progress_key = key
        result.update(self.clock_result(result))
        rate = ('waiting for contact/time' if result['final_score'] is None
                else f"{result['final_score']:.4f} points/s")
        rate_detail = (f" | RATE {result['rate_points']:.2f}/45 over {result['task_time_s']:.3f}s"
                       if 'rate_points' in result else '')
        print(f"[Evaluation] sim {result['elapsed_episode_time_s']:.2f}s | "
              f"contact +{result.get('contact_elapsed_time_s', result['task_time_s']):.2f}s | {items} | "
              f"TOTAL {result['raw_points']:.2f}/50 | rate {rate}{rate_detail}", flush=True)

    def _physics_step(self, dt, context=None):
        if not self.active:
            return
        try:
            if not np.isclose(float(dt), self.dt):
                raise ValueError('Physics time step changed during evaluation')
            self.ticks += 1
            physical = self.measurements.physical_sample()
            self.clock.update(self.ticks, self.measurements.contact(physical[0])
                              if self.clock.first_contact_tick is None else False)
            self.interval_holding = [a and b for a, b in zip(self.interval_holding, physical[1])]
            self.interval_clear = self.interval_clear and physical[2]
            self.interval_lifted = [a and b for a, b in zip(self.interval_lifted, physical[3])]
            if self.ticks % self.stride == 0:
                sample = self.measurements.measure(physical)
                sample['gripper_holding'] = self.interval_holding
                sample['garment_lifted_clear'] = self.interval_clear
                sample['gripper_lifted'] = self.interval_lifted
                self._record(sample)
                self.interval_holding, self.interval_clear = [True, True], True
                self.interval_lifted = [True, True]
        except Exception as exc:
            # Invalid evidence must not quietly yield a valid partial score.
            self.finish(reason=f'measurement_error: {exc}', valid=False, take_final=False)

    def finish(self, reason='user_stop', valid=True, take_final=True):
        if not self.active:
            return self.last_report
        self.active = False
        self.subscription = None
        try:
            if take_final and self.ticks * self.dt > self.scorer.last['time_s'] + 1e-8:
                sample = self.measurements.measure()
                sample['gripper_holding'] = [a and b for a, b in zip(self.interval_holding, sample['gripper_holding'])]
                sample['garment_lifted_clear'] &= self.interval_clear
                sample['gripper_lifted'] = [a and b for a, b in zip(self.interval_lifted, sample['gripper_lifted'])]
                self._record(sample)
            result = self.scorer.result()
            result.update(self.clock_result(result))
            detector = getattr(self.measurements, 'contact_detector', None)
            self.metadata['first_contact_collider'] = getattr(detector, 'last_collider', None)
            if 'recording_start_tick' in self.metadata:
                first = result['first_contact_tick']
                self.metadata['recording_first_contact_tick'] = None if first is None else self.metadata['recording_start_tick'] + first
        except Exception as exc:
            valid, result, reason = False, None, f'{reason}; {exc}'
        self.stream.close()
        report = {'schema_version': 'phase1-live-v4', 'valid': valid, 'reason': reason,
                  'metadata': self.metadata, 'result': result if valid else None}
        (self.path / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        self.last_report = report
        if valid:
            items = format_score_items(result)
            if (items, self.clock.first_contact_tick) != self._last_progress_key:
                print(f"[Evaluation] ITEMS: {items}", flush=True)
            rate = f"{result['final_score']:.6f} points/s" if result['final_score'] is not None else f"N/A ({result['score_status']})"
            rate_detail = (f"; numerator {result['rate_points']:.2f}/45, pickup excluded"
                           if 'rate_points' in result else '')
            print(f"[Evaluation] FINAL SCORE: {rate} "
                  f"({result['raw_points']:.2f}/50, {result['task_time_s']:.3f}s{rate_detail}) | {self.path / 'result.json'}", flush=True)
        else:
            print(f"[Evaluation] INVALID: {reason} | {self.path / 'result.json'}", flush=True)
        return report
