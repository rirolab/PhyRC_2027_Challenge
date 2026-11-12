"""Step-3 preflight diagnostics. Descent/retreat executors are pending."""
import math
from copy import deepcopy
import numpy as np


class Step3Shell:
    VERSION = 'step3-local-table-v3'
    PHASES = ('WAIT_STEP2', 'S3_PREFLIGHT', 'S3_HOLD', 'S3_LOWER',
              'S3_VERIFY', 'S3_RETREAT', 'S3_READY_FOR_INSERT', 'S3_STOP')

    def __init__(self):
        self.phase = 'WAIT_STEP2'
        self.started = None
        self.last_time = None
        self.stop_reason = None
        self.stop_time = None
        self.stop_snapshot = None
        self.latched_faults = []

    def tick(self, now, step2_ready, observations):
        previous = self.phase
        faults = []
        if not math.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            faults.append('S3_TIME_INVALID')
        else:
            self.last_time = now
        if len(observations) != 2:
            faults.append('S3_PAIR_INPUT_INVALID')
        if faults:
            self.phase = 'S3_STOP'
        elif self.phase != 'S3_STOP':
            if self.started is None and step2_ready:
                self.started = now
                self.phase = 'S3_PREFLIGHT'
            elif self.started is not None:
                self.phase = 'S3_HOLD'
                if now - self.started >= 12.:
                    self.phase = 'S3_STOP'
                    faults.append('S3_PREFLIGHT_TIMEOUT')
        robots = []
        for robot, observation in enumerate(observations):
            blockers = []
            if not step2_ready:
                blockers.append('S3_STEP2_NOT_READY')
            for key, code in (
                ('region_visible', 'S3_REGION_MISSING'),
                ('plane_fresh', 'S3_PLANE_STALE'),
                ('pose_ok', 'S3_POSE_OUTSIDE_STAGING'),
                ('pair_clear', 'S3_PAIR_GUARD')):
                if not observation.get(key, False):
                    blockers.append(code)
            # None means this measurement/certificate has not been implemented.
            for key, code in (
                ('local_table_verified', 'S3_LOCAL_TABLE_UNVERIFIED'),
                ('swept_volume_verified', 'S3_SWEEP_UNVERIFIED'),
                ('stopping_distance_verified', 'S3_STOPPING_DISTANCE_UNVERIFIED')):
                if observation.get(key) is not True:
                    blockers.append(code)
            height = observation.get('hover_clearance_m')
            if height is None or not math.isfinite(height):
                blockers.append('S3_CLEARANCE_UNAVAILABLE')
            elif height < .080:
                blockers.append('S3_ENTRY_TOO_LOW')
            blockers.append('S3_EXECUTOR_PENDING')
            robots.append(dict(robot=robot, blockers=blockers,
                               measured=deepcopy(observation),
                               evidence_status={key: ('UNIMPLEMENTED' if observation.get(key) is None
                                   else 'VERIFIED' if observation.get(key) is True else 'REJECTED')
                                   for key in ('local_table_verified', 'swept_volume_verified',
                                               'stopping_distance_verified')}))
        # Preserve the first stop and its evidence. Later observations remain
        # visible under robots/current_faults without rewriting history.
        if self.phase == 'S3_STOP' and self.stop_snapshot is None:
            self.stop_reason = faults[0] if faults else 'S3_STOP_UNSPECIFIED'
            self.stop_time = now if math.isfinite(now) else None
            self.stop_snapshot = deepcopy(dict(step2_ready=bool(step2_ready), robots=robots,
                                               faults=faults, previous_phase=previous))
        for fault in faults:
            if fault not in self.latched_faults:
                self.latched_faults.append(fault)
        return dict(phase=self.phase, previous_phase=previous,
                    version=self.VERSION,
                    event=previous != self.phase, step2_ready=bool(step2_ready),
                    step3_complete=False, descent_authorized=False,
                    retreat_authorized=False, faults=self.latched_faults.copy(),
                    current_faults=faults, stop_reason=self.stop_reason,
                    stop_time_s=self.stop_time, stop_snapshot=deepcopy(self.stop_snapshot),
                    stop_age_s=(max(0., now-self.stop_time) if self.stop_time is not None
                                and math.isfinite(now) else None),
                    robots=robots, descent_enabled=False,
                    mode='diagnostic_shell')

    def observe_servo(self, servo):
        observations = []
        reports = servo.alignment_report or []
        age = servo.t - getattr(servo, 'alignment_report_time', -1e9)
        fresh = 0. <= age <= .15
        for robot in (0, 1):
            report = reports[robot] if len(reports) == 2 and fresh else {}
            observations.append(dict(
                region_visible=servo.patch_debug[robot].get('outcome') == 'STAGING_REGION_VISIBLE'
                    and not (robot in servo.diag.get('align_evidence', {}).get('missing_patch_robots', [])),
                plane_fresh=bool(servo.diag.get('align_evidence', {}).get('plane_fresh', False)),
                pose_ok=bool(fresh and report and not report.get('blockers')),
                pair_clear=bool(not servo.diag.get('base_guard_active', True) and servo.fault is None),
                hover_clearance_m=report.get('hover_clearance_m'),
                staging_report_fresh=bool(fresh),
                staging_blockers=list(report.get('blockers', [])),
                region_outcome=servo.patch_debug[robot].get('outcome'),
                servo_fault=servo.fault,
                local_table_verified=None, swept_volume_verified=None,
                stopping_distance_verified=None))
        if servo.phase == 'ALIGNED' and servo.finished:
            bodies = [servo.bodies(r) for r in (0, 1)] if servo.plane is not None else None
            for robot in (0, 1):
                result = observe_local_table(servo, robot, bodies)
                observations[robot]['local_table_verified'] = result['verified']
                observations[robot]['local_table'] = result
        return self.tick(servo.t, servo.phase == 'ALIGNED' and servo.finished, observations)


TABLE_SUPPORT_RADIUS_M = .015
TABLE_PLANE_BAND_M = .008
TABLE_MIN_POINTS = 12
TABLE_LOG_SAMPLE_LIMIT = 4


def nearest_table_samples(support, queries, normal, offset):
    """Return usable sample count and nearest planar distance per query."""
    support = np.asarray(support, dtype=float).reshape(-1, 3)
    support = support[np.all(np.isfinite(support), axis=1)]
    heights = np.sum(support * normal, axis=1)
    support = support[np.abs(heights + offset) <= TABLE_PLANE_BAND_M]
    if not len(support):
        return 0, np.full(len(queries), np.inf)

    # Project into two orthonormal axes on the table for metric distances.
    tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    points_2d = np.column_stack([np.sum(support * axis, axis=1)
                                 for axis in (tangent, bitangent)])
    queries_2d = queries @ np.column_stack([tangent, bitangent])
    x, y = points_2d.T
    nearest = [np.sqrt(np.min((x-qx)**2 + (y-qy)**2)) for qx, qy in queries_2d]
    return len(support), np.asarray(nearest)


def check_table_coverage(support, queries, normal, offset, plane_age, *,
                         head_support=(), partner_head_support=()):
    """Fuse current observed samples; unobserved queries remain unknown."""
    result = dict(verified=False, reason=None, support_points=0, query_count=0,
                  uncovered_count=None, uncovered_indices=[], max_nearest_m=None)
    queries = np.asarray(queries, dtype=float).reshape(-1, 3)
    normal = np.asarray(normal, dtype=float)
    if (normal.shape != (3,) or not np.all(np.isfinite(normal))
            or abs(np.linalg.norm(normal)-1.) > .001 or not np.isfinite(offset)):
        result['reason'] = 'S3_TABLE_PLANE_INVALID'
        return result
    if not np.isfinite(plane_age) or not 0 <= plane_age <= 1.:
        result['reason'] = 'S3_TABLE_PLANE_STALE'
        return result
    if not len(queries) or not np.all(np.isfinite(queries)):
        result['reason'] = 'S3_TABLE_FOOTPRINT_INVALID'
        return result

    wrist_count, wrist_nearest = nearest_table_samples(support, queries, normal, offset)
    head_count, head_nearest = nearest_table_samples(head_support, queries, normal, offset)
    partner_count, partner_nearest = nearest_table_samples(partner_head_support, queries, normal, offset)
    partner_seen = partner_nearest <= TABLE_SUPPORT_RADIUS_M
    wrist_seen = wrist_nearest <= TABLE_SUPPORT_RADIUS_M
    head_seen = head_nearest <= TABLE_SUPPORT_RADIUS_M
    nearest = np.minimum(np.minimum(wrist_nearest, head_nearest), partner_nearest)
    missing = np.flatnonzero(~(wrist_seen | head_seen | partner_seen))
    result.update(
        support_points=wrist_count + head_count + partner_count, query_count=len(queries),
        wrist_support_points=wrist_count, head_support_points=head_count,
        wrist_covered_count=int(wrist_seen.sum()), head_covered_count=int(head_seen.sum()),
        head_added_count=int((head_seen & ~wrist_seen).sum()),
        partner_head_support_points=partner_count,
        partner_head_covered_count=int(partner_seen.sum()),
        partner_head_added_count=int((partner_seen & ~(wrist_seen | head_seen)).sum()),
        uncovered_count=len(missing), uncovered_indices=missing.tolist(),
        max_nearest_m=float(nearest.max()) if np.all(np.isfinite(nearest)) else None,
    )
    if result['support_points'] < TABLE_MIN_POINTS:
        result['reason'] = 'S3_TABLE_POINTS_INSUFFICIENT'
    else:
        result['verified'] = not len(missing)
        result['reason'] = 'S3_TABLE_OBSERVED' if result['verified'] else 'S3_TABLE_COVERAGE_GAP'
    return result


def table_cloud(servo, robot, camera_index, boxes):
    from rgbd_servo import geometry_masks, dilate

    image = np.asarray(servo.obs['rgb'])[camera_index]
    blue, yellow = geometry_masks(image, servo.masks)
    neutral = (np.ptp(image.astype(float), axis=2) < 35) & (image.mean(axis=2) > 50)
    mask = neutral & ~dilate(blue | yellow, 3)
    support = servo.cloud(camera_index, servo.geo[robot], mask, limit=65536)
    for box in boxes:
        inside = np.all((support >= box.min(0)-.005) & (support <= box.max(0)+.005), axis=1)
        support = support[~inside]
    return support


def observe_local_table(servo, robot, bodies):
    """Project each fingertip visual envelope along world lift Z."""
    if servo.plane is None:
        return dict(verified=False, reason='S3_TABLE_PLANE_MISSING')
    normal, offset, _ = servo.plane
    normal = np.asarray(normal, float)
    if (normal.shape != (3,) or not np.all(np.isfinite(normal)) or normal[2] < .9
            or abs(np.linalg.norm(normal)-1.) > .001 or not np.isfinite(offset)):
        return dict(verified=False, reason='S3_TABLE_PLANE_INVALID')
    plane_age = servo.t - servo.plane_time
    if not np.isfinite(plane_age) or not 0 <= plane_age <= 1.:
        return dict(verified=False, reason='S3_TABLE_PLANE_STALE')
    if bodies is None or len(bodies) != 2:
        return dict(verified=False, reason='S3_TABLE_FOOTPRINT_INVALID')
    for gripper in bodies:
        if any('gripper_fingertip_'+side+'_link' not in gripper for side in ('left', 'right')):
            return dict(verified=False, reason='S3_TABLE_FOOTPRINT_INVALID')
        for box in gripper.values():
            points = np.asarray(box)
            if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                    or not np.all(np.isfinite(points))):
                return dict(verified=False, reason='S3_TABLE_FOOTPRINT_INVALID')

    wrist_index, head_index = 1 + robot, 3 + robot
    boxes = [box for gripper in bodies for box in gripper.values()]
    wrist = table_cloud(servo, robot, wrist_index, boxes)
    head = table_cloud(servo, robot, head_index, boxes)
    partner_index = 4 - robot
    partner = table_cloud(servo, 1-robot, partner_index, boxes)
    footprints = []
    for side in ('left', 'right'):
        tip = bodies[robot]['gripper_fingertip_'+side+'_link']
        low, high = tip[:, :2].min(0)-.005, tip[:, :2].max(0)+.005
        axes = [np.linspace(a, b, max(2, int(np.ceil((b-a)/.01))+1))
                for a, b in zip(low, high)]
        queries = np.array([[x, y, -(normal[0]*x+normal[1]*y+offset)/normal[2]]
                            for x in axes[0] for y in axes[1]])
        footprints.append((side, low, high, queries))

    diagnostics = {}
    heights = {}
    for side, low, high, queries in footprints:
        tip = bodies[robot]['gripper_fingertip_'+side+'_link']
        heights[side] = float(np.min(tip @ normal + offset))
        coverage = check_table_coverage(
            wrist, queries, normal, offset, plane_age,
            head_support=head, partner_head_support=partner,
        )
        missing = np.asarray(coverage.pop('uncovered_indices'), dtype=int)
        coverage.update(
            min_clearance_m=heights[side],
            uncovered_world_examples=queries[missing[:TABLE_LOG_SAMPLE_LIMIT]].round(4).tolist(),
            footprint_min_xy=low.round(4).tolist(), footprint_max_xy=high.round(4).tolist(),
        )
        diagnostics[side] = coverage

    # Near-equal low surfaces both need coverage. Avoid arbitrary left/right tie.
    lower = min(heights, key=heights.get)
    required = [side for side in heights if heights[side]-heights[lower] <= .005]
    verified = all(diagnostics[side]['verified'] for side in required)
    result = dict(
        verified=verified, reason='S3_TABLE_LOWER_OBSERVED' if verified else 'S3_TABLE_LOWER_UNVERIFIED',
        lower_jaw=lower, required_jaws=required,
        coverage_scope='current_lower_fingertip_footprint',
        upper_clearance_verification_pending=True,
        wrist_camera=wrist_index, head_camera=head_index,
        partner_head_camera=partner_index, partner_head_owner=1-robot,
        time=servo.t, source='current_wrist_own_head_partner_head_rgbd',
        plane_age_s=plane_age, fingertips=diagnostics,
    )
    return result
