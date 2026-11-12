"""Episode-local public-pose traversal evidence; never a collision certificate.

The full-rate base pose stream is recorded independently of camera map_stride.
Swept disks are a geometric approximation of the base footprint. They establish
WHERE the base was observed, not that the floor/support/3D clearance is safe.
Invalid, discontinuous and implausibly large pose jumps are never bridged.
"""
from __future__ import annotations
import math
import numpy as np


class TraversalMemory:
    def __init__(self, origin_xy, resolution, shape, *, base_radius_m=.215,
                 max_segment_m=.30, max_elapsed_s=.75):
        self.origin = np.asarray(origin_xy, float).reshape(2)
        self.resolution = float(resolution)
        self.shape = tuple(int(x) for x in shape)
        self.base_radius_m = float(base_radius_m)
        self.max_segment_m = float(max_segment_m)
        self.max_elapsed_s = float(max_elapsed_s)
        if (self.resolution <= 0 or len(self.shape) != 2 or min(self.shape) < 1
                or self.base_radius_m <= 0 or self.max_segment_m <= 0 or self.max_elapsed_s <= 0):
            raise ValueError('invalid traversal geometry or continuity settings')
        self.base_visit_count_2d = np.zeros(self.shape, np.uint16)
        self.centerline_count_by_robot = np.zeros((2, *self.shape), np.uint16)
        self.footprint_count_by_robot = np.zeros((2, *self.shape), np.uint16)
        self.frames = []
        self.times = []
        self.base_poses = []
        self.segment_continuity = []  # 0 new/unknown; 1 continuous; 2 discontinuity; 3 invalid
        self.discontinuities = 0
        self.last = None
        self.last_frame = -1
        self.last_time = float('-inf')

    def _cell(self, pt):
        return np.floor((np.asarray(pt) - self.origin) / self.resolution).astype(int)

    def _mark_center(self, robot, pt):
        xy = self._cell(pt)
        if np.all(xy >= 0) and np.all(xy < self.shape):
            i, j = map(int, xy)
            self.centerline_count_by_robot[robot, i, j] = min(65535, int(self.centerline_count_by_robot[robot, i, j]) + 1)

    def _mark_footprint(self, robot, pt):
        # Using cell centers and an inscribed/circumscribed approximation is only
        # a VISIT TRACE. This mask is never used to certify free voxels.
        low = np.maximum(0, np.floor((pt - self.base_radius_m - self.origin) / self.resolution).astype(int))
        high = np.minimum(self.shape, np.ceil((pt + self.base_radius_m - self.origin) / self.resolution).astype(int) + 1)
        if np.any(high <= low):
            return
        xx, yy = np.meshgrid(np.arange(low[0], high[0]), np.arange(low[1], high[1]), indexing='ij')
        centers = self.origin + (np.stack((xx, yy), axis=-1) + .5) * self.resolution
        inside = np.sum((centers - pt) ** 2, axis=-1) <= self.base_radius_m ** 2
        vals = self.footprint_count_by_robot[robot, low[0]:high[0], low[1]:high[1]]
        if inside.any():
            vals[inside] = np.minimum(vals[inside].astype(np.uint32) + 1, 65535).astype(np.uint16)

    def observe(self, frame, time_s, base_pose_world):
        frame = int(frame)
        t = float(time_s)
        if frame <= self.last_frame or not np.isfinite(t) or t <= self.last_time:
            raise ValueError('traversal observation timestamps must increase')
        poses = np.asarray(base_pose_world, float)
        if poses.shape != (2, 7):
            raise ValueError('expected public base_pose_world with shape (2,7)')
        good = np.isfinite(poses).all(axis=1) & (np.linalg.norm(np.nan_to_num(poses[:, 3:]), axis=1) > .5)
        status = np.zeros(2, np.uint8)
        self.frames.append(frame)
        self.times.append(t)
        self.base_poses.append(poses.copy())
        for robot in range(2):
            if not good[robot]:
                status[robot] = 3
                continue
            p = poses[robot, :2]
            self._mark_footprint(robot, p)
            self._mark_center(robot, p)
            ij = self._cell(p)
            if np.all(ij >= 0) and np.all(ij < self.shape):
                i, j = map(int, ij)
                self.base_visit_count_2d[i, j] = min(65535, int(self.base_visit_count_2d[i, j]) + 1)
            if self.last is None or not self.last['good'][robot]:
                continue
            prev = self.last['poses'][robot, :2]
            displacement = float(np.linalg.norm(p - prev))
            dt = t - self.last['time_s']
            if dt > self.max_elapsed_s or displacement > self.max_segment_m:
                status[robot] = 2
                self.discontinuities += 1
                continue
            status[robot] = 1
            if displacement > 0:
                segments = max(1, int(math.ceil(displacement / (self.resolution / 2))))
                for k in range(1, segments):
                    q = prev + (p - prev) * (k / segments)
                    self._mark_center(robot, q)
                    self._mark_footprint(robot, q)
        self.segment_continuity.append(status)
        self.last = {'poses': poses.copy(), 'good': good.copy(), 'time_s': t}
        self.last_frame = frame
        self.last_time = t
        return dict(frame=frame, segment_status=status.tolist(),
                    discontinuities=self.discontinuities,
                    sampled_poses=len(self.frames))

    def export(self):
        poses = np.stack(self.base_poses).astype(np.float32) if self.base_poses else np.empty((0, 2, 7), np.float32)
        return dict(
            base_pose_history=poses,
            base_pose_history_frames=np.asarray(self.frames, np.int32),
            base_pose_history_time_s=np.asarray(self.times, np.float64),
            traversal_segment_continuity=np.stack(self.segment_continuity).astype(np.uint8) if self.segment_continuity else np.empty((0, 2), np.uint8),
            traversed_centerline_count_by_robot_2d=self.centerline_count_by_robot,
            traversed_footprint_count_by_robot_2d=self.footprint_count_by_robot,
            traversal_evidence_status=np.array('PUBLIC_POSE_TRACE_ONLY_NOT_COLLISION_CERTIFIED'),
            traversal_discontinuities=np.array(self.discontinuities, np.int32),
            traversal_base_radius_m=np.array(self.base_radius_m, np.float32),
        )
