"""Causal garment-to-robot proximity prior from public head-camera RGB-D.

This module makes *only a role proposal* (never a motion authorization).
RGB mask -> valid metric depth -> world frame must be done upstream by Mapper.
Never use overview camera or privileged simulator garment/scene coordinates.
Distances are evaluated at the same policy observation/frame as the point cloud.
"""
from __future__ import annotations

import math
import numpy as np


class GarmentProximityEstimator:
    """Freeze the earliest temporally corroborated, *pre-contact* garment-side vote.

    A single image, very small separation advantage, or disagreeing camera
    observations is insufficient. A successful choice is an unverified
    heuristic scene prior, not validated cloth identification/pose estimation.
    """

    def __init__(self, *, min_frames=2, min_margin_m=0.16,
                 max_center_shift_m=0.45, max_camera_disagreement_m=0.55):
        self.min_frames = int(min_frames)
        self.min_margin_m = float(min_margin_m)
        self.max_center_shift_m = float(max_center_shift_m)
        self.max_camera_disagreement_m = float(max_camera_disagreement_m)
        self.samples = []
        self.selection = None
        self.latest = dict(status='AWAITING_PREGRASP_RGBD', accepted_frames=0)

    def update(self, *, frame, head_evidence, base_pose_world):
        if self.selection is not None:
            return None
        candidates = []
        for item in head_evidence or []:
            try:
                if not str(item['camera']).endswith('_head') or int(item['points']) < 8:
                    continue
                p = np.asarray(item['garment_xyz_world'], dtype=float)
                if p.shape != (3,) or not np.isfinite(p).all():
                    continue
                candidates.append(p[:2])
            except (KeyError, ValueError, TypeError):
                continue
        if not candidates:
            self.latest = dict(status='NO_VALID_RGBD_CLOTH', frame=int(frame),
                               accepted_frames=len(self.samples))
            return None
        points = np.array(candidates)
        if len(points)>1 and np.max(np.linalg.norm(points-points[0],axis=1))>self.max_camera_disagreement_m:
            self.latest = dict(status='HEAD_CAMERAS_DISAGREE', frame=int(frame),
                               accepted_frames=len(self.samples))
            return None
        garment_xy = np.median(points,axis=0)
        try:
            robot_xy = np.asarray(base_pose_world,dtype=float).reshape(2,7)[:,:2]
        except (ValueError,TypeError):
            self.latest = dict(status='INVALID_BASE_POSES', frame=int(frame))
            return None
        if not np.isfinite(robot_xy).all():
            self.latest = dict(status='INVALID_BASE_POSES', frame=int(frame))
            return None
        d = np.linalg.norm(robot_xy-garment_xy,axis=1)
        margin = float(abs(d[0]-d[1]))
        if margin < self.min_margin_m:
            self.latest = dict(status='DISTANCE_TIE', frame=int(frame),
                               distance_r0_m=float(d[0]), distance_r1_m=float(d[1]),
                               accepted_frames=len(self.samples))
            return None
        near = int(np.argmin(d))
        sample = dict(frame=int(frame),closest_robot=near,
                      garment_xy_world=garment_xy.tolist(),
                      distance_r0_m=float(d[0]),distance_r1_m=float(d[1]),
                      margin_m=margin,head_cameras=len(candidates))
        if self.samples:
            recent=self.samples[-1]
            shift=float(np.linalg.norm(garment_xy-np.asarray(recent['garment_xy_world'])))
            if shift>self.max_center_shift_m or near != recent['closest_robot']:
                # Do not average two conflicting or significantly displaced
                # cloth observations. Restart the short corroboration window.
                self.samples.clear()
        self.samples.append(sample)
        self.samples=self.samples[-self.min_frames:]
        self.latest=dict(status='COLLECTING_EVIDENCE',frame=int(frame),
                         accepted_frames=len(self.samples),**{k:v for k,v in sample.items() if k!='frame'})
        if len(self.samples)<self.min_frames:
            return None
        if len({s['frame'] for s in self.samples})<self.min_frames:
            return None
        if len({s['closest_robot'] for s in self.samples})!=1:
            return None
        center=np.median([s['garment_xy_world'] for s in self.samples],axis=0)
        self.selection=dict(status='PROXIMITY_ROLE_HEURISTIC',
                            closest_robot=near, yielding_robot=1-near,
                            passing_robot=near,
                            garment_xy_world=center.tolist(),
                            median_margin_m=float(np.median([s['margin_m'] for s in self.samples])),
                            evidence_frames=[s['frame'] for s in self.samples],
                            head_camera_observations=sum(s['head_cameras'] for s in self.samples),
                            physical_grasp_verified=False,
                            source='PUBLIC_HEAD_RGBD_COLOR_MASK_PREGRASP_AND_BASE_POSE_WORLD',
                            geometry_verified=False,
                            motion_authorized=False)
        self.latest=self.selection.copy()
        return self.selection.copy()
