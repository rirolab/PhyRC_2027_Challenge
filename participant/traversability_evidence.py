"""Observation-grounded, NEVER motion-certified RGB-D traversability review.

Input layers already come from public-camera RGB-D and the existing conservative
3D occupancy pipeline. This module neither changes occupancy probabilities nor
marks unknown cells FREE in the strict map. It only narrows the previous broad
what-if review to measured floor + observed free rays, rejecting detected obstacles.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve, distance_transform_edt


def build_grounded_evidence(strict, review, *, min_floor_votes=2, min_free_layers=1,
                            base_visits=None):
    """Build an evidence-gated candidate map for diagnostic searches only.

    Tier 1: floor votes >=2 AND at least one known-free voxel in base height band.
    Tier 2: a one-cell measured-floor hole surrounded by >=6 grounded neighbors,
            AND at least one free voxel. Tier 2 is interpolation, not measurement.
    No seen floor: NOT admitted. Unknown voxel heights remain unknown; the whole
    robot column, arms, garment and dynamic obstacles are NOT certified.
    """
    strict_cost = np.asarray(strict['base_cost_2d'], np.uint8)
    review_cost = np.asarray(review['base_review_cost_2d'], np.uint8)
    floors = np.asarray(review['base_review_floor_votes_2d'])
    free_layers = np.asarray(review['base_review_observed_free_2d'])
    obstacle = np.asarray(review['base_review_obstacle_2d'], bool)
    if strict_cost.ndim != 2 or any(x.shape != strict_cost.shape for x in
            (review_cost, floors, free_layers, obstacle)):
        raise ValueError('incompatible RGB-D review grid shapes')
    if min_floor_votes < 1 or min_free_layers < 1:
        raise ValueError('evidence thresholds must be positive')
    measured_floor = floors >= min_floor_votes
    nearby = convolve(measured_floor.astype(np.uint8), np.ones((3, 3), np.uint8),
                      mode='constant', cval=0)
    # Local 5cm grid holes only; no extrapolation through occlusion or across rooms.
    interpolated_floor = (~measured_floor) & (nearby >= 6) & (floors == 0)
    ray_evidence = free_layers >= min_free_layers
    physically_clear_hypothesis = (review_cost < 255) & (~obstacle)
    grounded = measured_floor & ray_evidence & physically_clear_hypothesis
    interpolated = interpolated_floor & ray_evidence & physically_clear_hypothesis
    certified_by_old_strict = strict_cost < 255
    cost = np.full(strict_cost.shape, 255, dtype=np.uint8)
    cost[interpolated] = 210
    cost[grounded] = 168
    cost[grounded & (floors >= 4) & (free_layers >= 2)] = 140
    cost[certified_by_old_strict] = strict_cost[certified_by_old_strict]
    if base_visits is None:
        visits = np.zeros(strict_cost.shape, np.uint16)
        visits_available = False
    else:
        visits = np.asarray(base_visits)
        if visits.shape != strict_cost.shape:
            raise ValueError('base visit grid shape mismatch')
        visits_available = True
    tiers = np.zeros(strict_cost.shape, np.uint8)  # 0 no evidence
    tiers[measured_floor] = 1                         # 1 measured floor, missing ray or clearance
    tiers[interpolated] = 2                           # 2 interpolated floor (unverified)
    tiers[grounded] = 3                               # 3 measured floor + free ray (unverified)
    tiers[certified_by_old_strict] = 4                # 4 strict-existing
    tiers[obstacle] = 5                               # 5 detected obstacle
    return dict(
        grounded_cost_2d=cost,
        grounded_measured_floor_2d=measured_floor,
        grounded_interpolated_floor_2d=interpolated,
        grounded_ray_free_2d=ray_evidence,
        grounded_candidate_2d=(cost < 255),
        grounded_unverified_2d=(cost < 255) & (~certified_by_old_strict),
        grounded_evidence_tier_2d=tiers,
        grounded_base_visit_count_2d=visits,
        base_history_available=visits_available,
        status='UNVERIFIED_GEOMETRIC_EVIDENCE_PLAN_ONLY_NOT_FOR_EXECUTION',
    )


def start_causal_audit(xy, origin, resolution, strict, review, grounded,
                       *, strict_floor_min=None, height_unknown_count=None):
    """Explain WHY strict rejects each start (not just return cost=255)."""
    positions = np.asarray(xy, float).reshape(-1, 2)
    origin = np.asarray(origin, float).reshape(2)
    cost = np.asarray(strict['base_cost_2d'])
    floor = np.asarray(review['base_review_floor_votes_2d'])
    ray = np.asarray(review['base_review_observed_free_2d'])
    obstacle = np.asarray(review['base_review_obstacle_2d'])
    radius_margin = .215 + .02
    bounds = origin + np.asarray(cost.shape) * resolution
    floor_anchors = floor >= 2
    floor_dist = (distance_transform_edt(~floor_anchors) * resolution if floor_anchors.any()
                  else np.full(cost.shape, float('inf')))
    result=[]
    for robot, pt in enumerate(positions):
        ij = np.floor((pt-origin)/resolution).astype(int)
        inside=bool(np.all(ij>=0) and np.all(ij<cost.shape))
        info=dict(robot=robot,xy=pt.tolist(),cell=ij.tolist(),inside_map=inside,
                  nearest_map_edge_m=float(np.min(np.r_[pt-origin,bounds-pt])),
                  required_center_edge_m=radius_margin)
        if not inside:
            info['reasons']=['OUTSIDE_GRID'];result.append(info);continue
        idx=tuple(ij)
        st=int(cost[idx]); rv=int(review['base_review_cost_2d'][idx]);gr=int(grounded['grounded_cost_2d'][idx])
        unknown_z=(None if height_unknown_count is None else int(height_unknown_count[idx]))
        floor_strict=(None if strict_floor_min is None else int(strict_floor_min[idx]))
        reasons=[]
        if info['nearest_map_edge_m'] < radius_margin:
            reasons.append('FOOTPRINT_TOO_CLOSE_TO_MAP_EDGE')
        if bool(obstacle[idx]):reasons.append('OBSERVED_OBSTACLE_AT_CELL')
        if float(review['base_review_clearance_m'][idx]) <= 0:
            reasons.append('INSUFFICIENT_OBSTACLE_CLEARANCE')
        if int(floor[idx]) < 2:reasons.append('MEASURED_FLOOR_VOTES_BELOW_2')
        if floor_strict is not None and floor_strict < 2:
            reasons.append('STRICT_FLOOR_4_SUBCELLS_NOT_ALL_CONFIRMED')
        if int(ray[idx]) == 0:reasons.append('NO_OBSERVED_FREE_VOXEL_IN_BASE_HEIGHT_BAND')
        if unknown_z is not None and unknown_z > 0:
            reasons.append('UNOBSERVED_HEIGHT_VOXELS_IN_BASE_BAND')
        if st==255 and not reasons:reasons.append('FOOTPRINT_OR_OTHER_STRICT_CLEARANCE_REJECTION')
        info.update(strict_cost=st,review_cost=rv,grounded_cost=gr,
                    floor_votes=int(floor[idx]),free_height_layers=int(ray[idx]),
                    strict_floor_subcell_min_votes=floor_strict,
                    unknown_base_height_layers=unknown_z,
                    floor_anchor_distance_m=float(floor_dist[idx]),
                    base_visit_count=int(grounded['grounded_base_visit_count_2d'][idx]),
                    reasons=reasons)
        result.append(info)
    return result


def path_evidence_audit(waypoints, origin, resolution, strict, review, grounded):
    """Sample every <=half cell of a paired path, no action execution."""
    path = np.asarray(waypoints, float)
    if path.ndim != 3 or path.shape[1:] != (2, 2) or not len(path):
        return dict(status='NO_PATH')
    origin=np.asarray(origin,float)
    shape=np.asarray(grounded['grounded_cost_2d'].shape)
    points=[]
    for a,b in zip(path[:-1],path[1:]):
        n=max(1,int(np.ceil(float(np.linalg.norm(b-a,axis=1).max())/(resolution/2))))
        points.extend([a+(b-a)*(k/n) for k in range(n)])
    points.append(path[-1])
    p=np.asarray(points).reshape(-1,2)
    ij=np.floor((p-origin)/resolution).astype(int)
    inside=((ij>=0)&(ij<shape)).all(axis=1)
    counts=dict(sampled_robot_centers=int(len(p)),outside=int((~inside).sum()))
    for name,mask in (
        ('strict_free',np.asarray(strict['base_cost_2d'])<255),
        ('broad_review_candidate',np.asarray(review['base_review_cost_2d'])<255),
        ('grounded_candidate',np.asarray(grounded['grounded_cost_2d'])<255),
        ('measured_floor',grounded['grounded_measured_floor_2d']),
        ('has_free_ray',grounded['grounded_ray_free_2d']),
        ('observed_obstacle',review['base_review_obstacle_2d']),
    ):
        counts[name+'_samples']=int(mask[tuple(ij[inside].T)].sum()) if inside.any() else 0
    n=max(1,len(p))
    counts['grounded_fraction']=round(counts['grounded_candidate_samples']/n,4)
    counts['strict_fraction']=round(counts['strict_free_samples']/n,4)
    counts['has_unverified_segments']=bool(counts['grounded_candidate_samples']<len(p))
    counts['status']='UNVERIFIED_PATH_AUDIT_ONLY'
    return counts


def fuse_traversal_with_rgbd(strict, review, grounded, *,
                             centerline_count_by_robot=None,
                             footprint_count_by_robot=None):
    """Fuse OBSERVED base traversal with RGB-D as an UNVERIFIED planning layer.

    The presence of a robot at a pose does not by itself prove collision-free
    support or arm/cloth clearance. Only observed centerlines enter the fused
    costmap; the much wider swept footprint is retained for diagnostics.
    RGB-D obstacle and clearance exclusions ALWAYS win. Neither the strict
    map nor voxel occupancy is modified. Not authorized for execution.
    """
    st = np.asarray(strict['base_cost_2d'], np.uint8)
    rv = np.asarray(review['base_review_cost_2d'], np.uint8)
    obstruction = np.asarray(review['base_review_obstacle_2d'], bool)
    base = np.asarray(grounded['grounded_cost_2d'], np.uint8)
    if st.ndim != 2 or any(x.shape != st.shape for x in (rv, obstruction, base)):
        raise ValueError('strict/review/grounded map shape mismatch')
    if centerline_count_by_robot is None or footprint_count_by_robot is None:
        ctr = np.zeros((2, *st.shape), np.uint16)
        footprint = np.zeros_like(ctr)
        history = False
    else:
        ctr = np.asarray(centerline_count_by_robot, np.uint16)
        footprint = np.asarray(footprint_count_by_robot, np.uint16)
        if ctr.shape != (2, *st.shape) or footprint.shape != ctr.shape:
            raise ValueError('traversal trace grid shape mismatch')
        history = True
    centers = np.any(ctr > 0, axis=0)
    swept = np.any(footprint > 0, axis=0)
    # The review collision/clearance filter is a conservative denial, not a
    # proof of FREE. Disagreements are reported, never erased by traversal.
    denied = obstruction | (rv == 255)
    admitted = centers & ~denied
    conflict = centers & denied
    fused = base.copy()
    fused[admitted] = np.minimum(fused[admitted], 190)
    fused[st < 255] = st[st < 255]
    fused[denied] = 255
    return dict(
        traversal_fused_cost_2d=fused,
        traversal_centerline_candidate_2d=admitted,
        traversal_observed_centerline_2d=centers,
        traversal_observed_footprint_2d=swept,
        traversal_blocked_by_rgbd_2d=conflict,
        traversal_unverified_2d=(fused < 255) & (st == 255),
        traversal_history_present=history,
        traversal_evidence_status='PUBLIC_POSE_TRACE_AND_RGBD_REVIEW_NOT_COLLISION_CERTIFIED',
    )
