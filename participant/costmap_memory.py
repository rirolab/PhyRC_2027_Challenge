"""Episode-local RGB-D occupancy and conservative 2D feasibility primitives.

This file contains no Isaac Sim dependency. UNKNOWN is not FREE. Base-only geometry
must never be reported as certification of arm, cloth, IK, or dynamic clearance.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter

UNKNOWN = "UNKNOWN"
TRUE = "TRUE"
FALSE = "FALSE"


@dataclass(frozen=True)
class GridSpec:
    origin: tuple = (-3.5, -3.5, 0.0)
    resolution: float = 0.10
    shape: tuple = (70, 70, 24)  # 7 x 7 x 2.4 m, separate from the legacy 5 cm evidence grid

    def __post_init__(self):
        if self.resolution <= 0 or len(self.shape) != 3 or any(s < 1 for s in self.shape):
            raise ValueError("invalid 3D grid specification")


class OccupancyMemory:
    """Log-odds from static depth endpoints + truncated free-space rays.

    One update per voxel per camera per frame, not one update per pixel. Moving
    robots and garment pixels are excluded by the calling mapper. No invalid
    depth return generates free-space evidence. No ray extends beyond its hit.
    """
    def __init__(self, spec=None):
        self.spec = spec or GridSpec()
        sh = self.spec.shape
        self.log_odds = np.zeros(sh, dtype=np.float32)
        self.observation_count = np.zeros(sh, dtype=np.uint32)
        self.last_observed_time = np.full(sh, -1., dtype=np.float32)
        self.camera_mask = np.zeros(sh, dtype=np.uint8)
        self.view_sector_mask = np.zeros(sh, dtype=np.uint32)  # 8 azimuth x 3 elevation sectors
        self.conflicts = np.zeros(sh, dtype=np.uint16)
        self.num_updates = 0

    def _indices(self, xyz):
        p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        ijk = np.floor((p - np.asarray(self.spec.origin)) / self.spec.resolution).astype(np.int64)
        ok = ((ijk >= 0) & (ijk < np.asarray(self.spec.shape))).all(axis=1)
        return np.ravel_multi_index(ijk[ok].T, self.spec.shape) if ok.any() else np.empty(0, np.int64)

    def update_rays(self, origin, endpoints, camera_id, time_s, *, max_rays=4096):
        if not (0 <= int(camera_id) < 8):
            raise ValueError("camera_id must fit uint8 camera mask")
        if not np.isfinite(time_s):
            raise ValueError("nonfinite observation time")
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        pts = np.asarray(endpoints, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) > max_rays:
            pts = pts[np.linspace(0, len(pts)-1, max_rays).astype(int)]
        if not len(pts):
            return
        direction = pts - origin
        length = np.linalg.norm(direction, axis=1)
        good = length > self.spec.resolution
        direction, length, pts = direction[good], length[good], pts[good]
        if not len(pts):
            return
        direction /= length[:, None]
        step = self.spec.resolution * 0.65
        n = min(256, int(np.ceil(length.max() / step)))
        distances = (np.arange(1, n+1) * step).astype(np.float64)
        # Stop before the endpoint voxel, guarding against marking the hit as free.
        active = distances[None, :] < (length[:, None] - self.spec.resolution * 1.5)
        coords = origin + direction[:, None, :] * distances[None, :, None]
        free = np.unique(self._indices(coords[active]))
        hit = np.unique(self._indices(pts))
        # Conflicting evidence within one image is indeterminate for this update.
        both = np.intersect1d(free, hit, assume_unique=True)
        if len(both):
            free = np.setdiff1d(free, both, assume_unique=True)
            hit = np.setdiff1d(hit, both, assume_unique=True)
            cf = self.conflicts.ravel()
            cf[both] = np.minimum(65535, cf[both].astype(np.int64)+1)
        odds = self.log_odds.ravel()
        counts = self.observation_count.ravel()
        last = self.last_observed_time.ravel()
        cameras = self.camera_mask.ravel()
        sectors = self.view_sector_mask.ravel()
        for inds, delta in ((free, -0.48), (hit, 0.92)):
            if not len(inds):
                continue
            odds[inds] = np.clip(odds[inds] + delta, -5., 5.)
            counts[inds] = np.minimum(np.iinfo(np.uint32).max, counts[inds].astype(np.uint64)+1)
            last[inds] = float(time_s)
            cameras[inds] |= (1 << int(camera_id))
            ijk=np.column_stack(np.unravel_index(inds,self.spec.shape))
            centers=(ijk+.5)*self.spec.resolution + np.asarray(self.spec.origin)
            delta_pos=centers-origin
            az=np.clip(np.floor((np.arctan2(delta_pos[:,1],delta_pos[:,0])+np.pi)
                                  /(2*np.pi)*8).astype(int),0,7)
            elev=np.arctan2(delta_pos[:,2],np.linalg.norm(delta_pos[:,:2],axis=1)+1e-8)
            el=np.where(elev<-.25,0,np.where(elev>.25,2,1))
            bins=(az*3+el).astype(np.uint32)
            sectors[inds] |= np.left_shift(np.uint32(1),bins)
        if len(both):
            counts[both] += 1
            last[both] = float(time_s)
            cameras[both] |= (1 << int(camera_id))
        self.num_updates += 1

    def probability(self):
        return 1. / (1. + np.exp(-self.log_odds))

    def layers(self):
        observed = self.observation_count > 0
        p = self.probability()
        occupied = observed & (p >= .67)
        free = observed & (p <= .33)
        unknown = ~(occupied | free)
        n_cameras = np.zeros(self.spec.shape, np.uint8)
        for bit in range(8):
            n_cameras += (self.camera_mask & (1 << bit) != 0).astype(np.uint8)
        n_views = np.zeros(self.spec.shape, np.uint8)
        for bit in range(24):
            n_views += (self.view_sector_mask & (np.uint32(1) << np.uint32(bit)) != 0).astype(np.uint8)
        confidence = (1. - np.exp(-self.observation_count.astype(np.float32) / 3.)) * np.minimum(1., n_views / 2.)
        confidence[unknown] = 0.
        return dict(occupied=occupied, free=free, unknown=unknown,
                    confidence=confidence.astype(np.float32),
                    independent_camera_count=n_cameras,
                    independent_view_sector_count=n_views)

    def project_base(self, floor_support, *, output_res=.05, robot_radius=.215, margin=.02,
                     base_z_min=.10, base_z_max=.65):
        """Strict known-free base map. Unknown height cells remain impassable.

        For arbitrary oriented footprints, validate separately with footprint_check.
        """
        layer = self.layers()
        z = np.asarray(self.spec.origin)[2] + (np.arange(self.spec.shape[2]) + .5) * self.spec.resolution
        band = (z >= base_z_min) & (z <= base_z_max)
        if not np.any(band):
            raise ValueError("base height band does not intersect voxel map")
        obstacle = layer["occupied"][:, :, band].any(axis=2)
        unknown = layer["unknown"][:, :, band].any(axis=2)
        viewed = ~unknown & ~obstacle
        ratio = self.spec.resolution / output_res
        if abs(ratio-round(ratio)) > 1e-8:
            raise ValueError("output resolution must divide voxel resolution")
        r = int(round(ratio))
        obstacle = np.repeat(np.repeat(obstacle, r, 0), r, 1)
        unknown = np.repeat(np.repeat(unknown, r, 0), r, 1)
        floor_support = np.asarray(floor_support, bool)
        if floor_support.shape != obstacle.shape:
            raise ValueError("floor support shape incompatible with voxel grid")
        known_free = ~obstacle & ~unknown & floor_support
        # Pad outside of map with obstacle: distance at the edge is never infinite.
        distance = distance_transform_edt(np.pad(known_free, 1))[1:-1, 1:-1] * output_res
        clearance = distance - math.sqrt(2) * output_res / 2
        base_cost = np.full(known_free.shape, 255, np.uint8)
        feasible = known_free & (clearance > robot_radius + margin)
        base_cost[feasible] = np.clip(np.ceil(253*np.exp(-(clearance[feasible]-robot_radius-margin)/.2)), 1, 253).astype(np.uint8)
        return dict(base_cost_2d=base_cost, base_unknown_2d=unknown | ~floor_support,
                    base_occupied_2d=obstacle, base_known_free_2d=known_free,
                    base_clearance_m=(clearance-robot_radius-margin).astype(np.float32))

    def project_base_review(self, strict, floor_votes, near_floor_obstacle=None,
                            *, output_res=.05, robot_radius=.215, margin=.02,
                            base_z_min=.10, base_z_max=.65):
        """UNVERIFIED what-if 2D traversability from RGB-D geometric evidence.

        Strict `project_base` remains the only safety-oriented output. This
        speculative map is for plan_only/diagnostics and MUST NEVER control a
        robot. An unknown cell stays unknown in exported evidence even if a
        bounded-cost *hypothesis* lets a path search pass through its center.
        Occluded regions are NOT interpreted as measured free space.
        """
        if not isinstance(strict,dict) or 'base_cost_2d' not in strict:
            raise ValueError('strict occupancy projection required')
        layer=self.layers()
        z=self.spec.origin[2]+(np.arange(self.spec.shape[2])+.5)*self.spec.resolution
        band=(z>=base_z_min)&(z<=base_z_max)
        if not band.any():
            raise ValueError('empty base height band')
        ratio=self.spec.resolution/output_res
        if abs(ratio-round(ratio))>1e-8:
            raise ValueError('output_res must divide voxel resolution')
        r=int(round(ratio))
        def upsample(a):
            return np.repeat(np.repeat(a,r,axis=0),r,axis=1)
        obstacle=upsample(layer['occupied'][:,:,band].any(axis=2))
        observed_free=upsample(layer['free'][:,:,band].sum(axis=2))
        observed_hits=upsample(layer['occupied'][:,:,band].sum(axis=2))
        floor_votes=np.asarray(floor_votes)
        if floor_votes.shape!=obstacle.shape:
            raise ValueError('floor evidence incompatible with voxel projection')
        low=np.zeros_like(obstacle,dtype=bool) if near_floor_obstacle is None else np.asarray(near_floor_obstacle,bool)
        if low.shape!=obstacle.shape:
            raise ValueError('near-floor obstacle shape mismatch')
        obstacle=obstacle|low
        # Neither a map edge nor an observed obstacle is a traversable base center.
        free_distance=distance_transform_edt(np.pad(~obstacle,1,constant_values=False))[1:-1,1:-1]*output_res
        center_clearance=free_distance-math.sqrt(2)*output_res/2-robot_radius-margin
        possible=(center_clearance>0)&~obstacle
        # Floor evidence is kept as graded confidence, not a binary decree.
        # The max proximity (0.45m) is an explicit REVIEW heuristic, not proof
        # of unseen floor nor of free columns.
        anchors=floor_votes>=2
        floor_dist=(distance_transform_edt(~anchors)*output_res if anchors.any()
                    else np.full(obstacle.shape,float('inf'),dtype=np.float64))
        cost=np.full(obstacle.shape,255,np.uint8)
        risk=np.full(obstacle.shape,250,np.uint8)
        risk[observed_free>=1]=218
        risk[observed_free>=3]=191
        risk[floor_dist<=.45]=np.minimum(risk[floor_dist<=.45],226)
        risk[floor_dist<=.20]=np.minimum(risk[floor_dist<=.20],203)
        risk[anchors]=np.minimum(risk[anchors],177)
        risk[(anchors)&(observed_free>=1)]=132
        strict_cost=np.asarray(strict['base_cost_2d'],np.uint8)
        if strict_cost.shape!=cost.shape:
            raise ValueError('strict map has incompatible shape')
        verified=strict_cost<255
        risk[verified]=strict_cost[verified]
        cost[possible]=risk[possible]
        # Explicitly preserve independent uncertainty and method provenance.
        hypothetical=possible&(~verified)
        return dict(base_review_cost_2d=cost,
                    base_review_hypothesis_only_2d=hypothetical,
                    base_review_unobserved_floor_2d=floor_votes==0,
                    base_review_obstacle_2d=obstacle,
                    base_review_clearance_m=center_clearance.astype(np.float32),
                    base_review_observed_free_2d=observed_free.astype(np.uint8),
                    base_review_observed_hits_2d=observed_hits.astype(np.uint8),
                    base_review_floor_votes_2d=np.minimum(floor_votes,255).astype(np.uint8),
                    base_strict_cost_2d=strict_cost.copy(),
                    review_status='UNVERIFIED_HYPOTHESIS_PLAN_ONLY_NOT_SAFE_TO_EXECUTE')

    def export(self):
        ls = self.layers()
        return dict(voxel_log_odds=self.log_odds, voxel_observation_count=self.observation_count,
                    voxel_last_observed_time=self.last_observed_time,
                    voxel_camera_mask=self.camera_mask,
                    voxel_view_sector_mask=self.view_sector_mask,
                    voxel_camera_count=ls["independent_camera_count"],
                    voxel_view_count=ls["independent_view_sector_count"],
                    voxel_confidence=ls["confidence"], voxel_occupied=ls["occupied"],
                    voxel_free=ls["free"], voxel_unknown=ls["unknown"],
                    voxel_origin=np.asarray(self.spec.origin), voxel_resolution=self.spec.resolution,
                    voxel_shape=np.asarray(self.spec.shape), voxel_conflicts=self.conflicts)


def footprint_check(known_free, origin_xy, resolution, center, heading_rad,
                    length_m, width_m, margin_m=.02):
    """Conservative cell-center coverage of an oriented rectangular base footprint.

    Returns TRUE/FALSE/UNKNOWN; no arm or garment checks. Expand by a cell diagonal
    to prevent missing a cell on the rectangle boundary due to discretization.
    """
    known_free = np.asarray(known_free, bool)
    center = np.asarray(center, float)
    if known_free.ndim != 2 or center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("invalid footprint query")
    if min(length_m, width_m, resolution) <= 0 or margin_m < 0:
        raise ValueError("invalid footprint size")
    reach = .5 * math.hypot(length_m, width_m) + margin_m + resolution
    low = np.floor((center-reach-np.asarray(origin_xy))/resolution).astype(int)
    high = np.ceil((center+reach-np.asarray(origin_xy))/resolution).astype(int)
    if (low < 0).any() or (high >= np.asarray(known_free.shape)).any():
        return UNKNOWN
    ii,jj = np.mgrid[low[0]:high[0]+1, low[1]:high[1]+1]
    xyz = (np.stack((ii,jj),-1)+.5)*resolution + np.asarray(origin_xy)
    delta = xyz-center
    c,s = math.cos(heading_rad),math.sin(heading_rad)
    fwd = delta[...,0]*c + delta[...,1]*s
    side = -delta[...,0]*s + delta[...,1]*c
    pad = margin_m + resolution * math.sqrt(2)/2
    interior = (abs(fwd) <= length_m/2+pad) & (abs(side) <= width_m/2+pad)
    return TRUE if bool(known_free[ii[interior], jj[interior]].all()) else UNKNOWN


def swept_base_validation(cost, origin_xy, resolution, paths, *, radius=.215, margin=.02):
    """2D circle swept-edge conservative bound (not full 3D continuous collision)."""
    cost = np.asarray(cost)
    paths = np.asarray(paths, float)
    if paths.ndim != 3 or paths.shape[1:] != (2, 2):
        raise ValueError("paths must be [T,2,2]")
    if len(paths) == 0 or not np.isfinite(paths).all():
        return dict(status=UNKNOWN, reason="empty_or_invalid_path")
    allowed = cost < 255
    # Exact endpoint center is not enough: all subsegment samples must remain in mapped space.
    distance = distance_transform_edt(np.pad(allowed, 1))[1:-1, 1:-1] * resolution
    min_clear = float("inf")
    samples = 0
    for k in range(max(1, len(paths)-1)):
        p = paths[k]
        q = paths[min(k+1,len(paths)-1)]
        max_step = np.linalg.norm(q-p, axis=1).max()
        n = max(1, int(np.ceil(max_step/(resolution/2))))
        for alpha in np.linspace(0, 1, n+1):
            pose = (1-alpha)*p + alpha*q
            cell = np.floor((pose-np.asarray(origin_xy))/resolution).astype(int)
            if (cell < 0).any() or (cell >= np.asarray(cost.shape)).any():
                return dict(status=UNKNOWN, reason="outside_map", samples=samples)
            if not allowed[tuple(cell.T)].all():
                return dict(status=UNKNOWN, reason="occupied_or_unknown", samples=samples)
            # EDT measured to disallowed cell centers: conservative correction.
            lower = float(distance[tuple(cell.T)].min()) - math.sqrt(2)*resolution/2 - max_step/(2*n)
            min_clear = min(min_clear, lower-radius-margin)
            # Relative-distance Lipschitz bound between consecutive samples:
            # do not assert a safe inter-robot sweep from endpoints alone.
            relative_motion_bound = (np.linalg.norm(q[0]-p[0])+
                                     np.linalg.norm(q[1]-p[1]))/(2*n)
            if np.linalg.norm(pose[0]-pose[1]) - relative_motion_bound <= 2*(radius+margin):
                return dict(status=FALSE, reason="base_pair_overlap_or_unresolved_sweep", samples=samples)
            samples += 1
    return dict(status=TRUE if min_clear > 0 else UNKNOWN,
                reason="base_only_conservative_bound" if min_clear > 0 else "insufficient_clearance",
                minimum_clearance_lower_bound_m=min_clear, samples=samples,
                arm_status=UNKNOWN, garment_status=UNKNOWN,
                edge_validation_status="2D_conservative_sampled_bound_not_full_robot")


def assess_gap(known_free, obstacle, origin_xy, resolution, center, heading_rad,
               *, footprint_length=.43, footprint_width=.43, margin=.02, max_scan=1.5):
    """Measure one observed corridor cross-section; do not silently approve unknowns."""
    known_free = np.asarray(known_free, bool)
    obstacle = np.asarray(obstacle, bool)
    if known_free.shape != obstacle.shape:
        raise ValueError("grid shape mismatch")
    center = np.asarray(center, float)
    normal = np.array([-math.sin(heading_rad),math.cos(heading_rad)])
    bounds=[]
    unknown_edge=False
    for sign in (-1,1):
        edge=None
        for d in np.arange(0, max_scan+resolution/2, resolution/2):
            xy = center + sign*d*normal
            cell = np.floor((xy-np.asarray(origin_xy))/resolution).astype(int)
            if (cell < 0).any() or (cell >= np.asarray(known_free.shape)).any():
                unknown_edge=True;break
            ij=tuple(cell)
            if obstacle[ij]:
                edge=float(d);break
            if not known_free[ij]:
                unknown_edge=True;break
        bounds.append(edge)
    width = sum(bounds) if all(v is not None for v in bounds) else None
    oriented_width = abs(footprint_length*math.sin(heading_rad)) + abs(footprint_width*math.cos(heading_rad))
    usable = None if width is None else width-oriented_width-2*margin
    # Assess an entire short local passage, not just the center configuration.
    direction = np.array([math.cos(heading_rad),math.sin(heading_rad)])
    corridor_offsets = np.arange(-footprint_length/2,footprint_length/2+resolution/4,resolution/2)
    footprints=[footprint_check(known_free, origin_xy, resolution,center+a*direction,
                                heading_rad,footprint_length,footprint_width,margin)
                for a in corridor_offsets]
    passage_known=all(v==TRUE for v in footprints)
    if usable is not None and usable <= 0:
        status=FALSE
    elif unknown_edge or not passage_known or width is None:
        status=UNKNOWN
    else:
        status=TRUE
    return dict(center_world_xy=center.tolist(), heading_rad=float(heading_rad),
                observed_width_m=width, usable_width_m=usable,
                minimum_clearance_m=None if usable is None else usable/2,
                visibility_confidence=1. if not unknown_edge else 0.,
                base_passable=status, arm_passable=UNKNOWN,
                garment_passable=UNKNOWN, unknown_intersection=bool(unknown_edge or not passage_known),
                method="short_2D_local_corridor_not_full_path_or_3D_certification")


def detect_gaps(known_free, occupied, origin_xy, resolution, *, max_candidates=10,
                footprint_length=.43, footprint_width=.43, margin=.02):
    """Candidate medial corridors from observed geometry only, not validated full paths."""
    free = np.asarray(known_free,bool)
    obstacles = np.asarray(occupied,bool)
    if free.shape != obstacles.shape or not free.any() or not obstacles.any():
        return []
    dist = distance_transform_edt(free)*resolution
    # Local maxima of distance to nonfree cells approximate corridor centers.
    ridge = free & (dist==maximum_filter(dist, size=7)) & (dist > .15) & (dist < .9)
    points = np.argwhere(ridge)
    if not len(points):
        return []
    points = sorted(points.tolist(), key=lambda ij: (dist[tuple(ij)], ij[0],ij[1]))
    results=[]
    for ij in points:
        xy=(np.array(ij)+.5)*resolution + np.asarray(origin_xy)
        if any(np.linalg.norm(xy-np.asarray(c["center_world_xy"])) < .4 for c in results):
            continue
        for theta in (0., math.pi/4, math.pi/2, -math.pi/4):
            result=assess_gap(free, obstacles, origin_xy, resolution, xy,theta,
                              footprint_length=footprint_length,footprint_width=footprint_width,margin=margin)
            if result["observed_width_m"] is not None:
                result["geometry_class"]="narrow_passage_candidate"
                results.append(result)
                break
        if len(results)>=max_candidates:
            break
    return results


def sweep_coverage(cost, origin_xy, resolution, paths, *, max_step_fraction=.5):
    """Measure unknown/out-of-map exposure at sampled 2D base centers only."""
    cost=np.asarray(cost)
    paths=np.asarray(paths,float)
    if paths.ndim!=3 or paths.shape[1:]!=(2,2) or len(paths)<1:
        return dict(unknown_swept_fraction=None,sample_count=0,method="2D_samples")
    unknown=0;total=0
    for k in range(max(1,len(paths)-1)):
        p=paths[k];q=paths[min(k+1,len(paths)-1)]
        length=float(np.linalg.norm(q-p,axis=1).max())
        n=max(1,int(np.ceil(length/(resolution*max_step_fraction))))
        samples=p[None,:,:] + np.linspace(0,1,n+1)[:,None,None]*(q-p)[None,:,:]
        cells=np.floor((samples-np.asarray(origin_xy))/resolution).astype(int)
        inside=((cells>=0)&(cells<np.asarray(cost.shape))).all(axis=-1)
        ok=np.zeros_like(inside,bool)
        if inside.any():
            ok[inside]=(cost[tuple(cells[inside].T)]<255)
        unknown+=int((~ok).sum());total+=int(ok.size)
    return dict(unknown_swept_fraction=unknown/max(total,1),sample_count=total,
                method="2D_center_sampling_not_full_swept_volume")


def frontier_targets(known_free, unknown, origin_xy, resolution, start_xy, *, max_results=8):
    """Locate reachable observation frontiers (no actuator commands).

    Candidate cells must be currently known free, adjacent to unknown cells.
    This does not imply that a camera can actually see the unknown region.
    """
    from scipy.ndimage import binary_dilation
    free=np.asarray(known_free,bool)
    unknown=np.asarray(unknown,bool)
    if free.shape!=unknown.shape or not free.any() or not unknown.any():
        return []
    edge=free & binary_dilation(unknown,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],bool))
    cells=np.argwhere(edge)
    if not len(cells):return []
    start=np.asarray(start_xy,float).reshape(2)
    centers=(cells+.5)*resolution+np.asarray(origin_xy)
    dist=np.linalg.norm(centers-start,axis=1)
    # Deterministic ranking; distances are diagnostics, not certified routes.
    order=np.argsort(dist,kind="stable")[:max_results]
    return [dict(world_xy=centers[k].tolist(),distance_m=float(dist[k]),
                 status="VIEWPOINT_CANDIDATE_NOT_REACHABILITY_CERTIFIED") for k in order]
