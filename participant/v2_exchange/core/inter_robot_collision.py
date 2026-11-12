"""M2-C: reusable inter-robot USD mesh collision and sampled edge checking.

Frame: W_T_mesh = W_T_base @ base_T_link(q) @ link_T_mesh.
This is discrete sampling, NOT continuous/swept collision detection.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
from .forward_kinematics import read_usd_joints, forward_graph, mat, qrot
from .robot_meshes import ROOT, load_colliders


class InvalidConfiguration(ValueError):
    """Invalid shape, pose, unknown DOF, or joint-limit violation."""


@dataclass
class CollisionResult:
    collision_free: bool
    pairs: list
    # Structural checks are performed on construction; invalid inputs raise.


def _pose7(p):
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise InvalidConfiguration('Base pose must be finite shape (7,) [xyz,wxyz]')
    n = np.linalg.norm(p[3:])
    if n < 1e-12:
        raise InvalidConfiguration('Invalid zero quaternion')
    p = p.copy()
    p[3:] /= n
    return p


def _slerp(qa, qb, t):
    qa, qb = qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)
    dot = float(np.dot(qa, qb))
    if dot < 0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1., 1.))
    if dot > 0.9995:
        q = (1-t)*qa + t*qb
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    s = math.sin(theta)
    return (math.sin((1-t)*theta)/s)*qa + (math.sin(t*theta)/s)*qb


def interpolate_pose(a, b, t):
    a, b = _pose7(a), _pose7(b)
    return np.r_[(1-t)*a[:3]+t*b[:3], _slerp(a[3:], b[3:], t)]


def quaternion_angle(a, b):
    a, b = _pose7(a)[3:], _pose7(b)[3:]
    return 2 * math.acos(float(np.clip(abs(np.dot(a,b)), 0., 1.)))


class TwoRobotCollisionChecker:
    """Reusable FCL geometry managers for inter-robot collisions only.

    Not thread safe. Instantiate one checker per thread/process.
    Source mapping must correspond to HDF5 robot_0 (Stretch4), robot_1 (Stretch4_2).
    """
    def __init__(self, usd_path, mapping_path, joint_limit_tolerance=0.0):
        try:
            import trimesh
            import fcl  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('Install trimesh, python-fcl and scipy') from exc
        self.usd_path = str(usd_path)
        self.mapping = json.loads(Path(mapping_path).read_text())
        keys = ('/World/Stretch4', '/World/Stretch4_2')
        self.names = [self.mapping[k]['dof_names'] for k in keys]
        if len(self.names[0]) != 13 or self.names[0] != self.names[1] or len(set(self.names[0])) != 13:
            raise ValueError('Expected matching unique 13-DOF maps')
        for key, names in zip(keys, self.names):
            indices = self.mapping[key]['dof_indices']
            if any(indices[n] != i for i, n in enumerate(names)):
                raise ValueError(f'DOF indices inconsistent with names: {key}')
        self.joints = read_usd_joints(self.usd_path)
        self.colliders, unresolved, count = load_colliders(self.usd_path, self.joints)
        if unresolved or len(self.colliders) != count:
            raise RuntimeError('Incomplete collision geometry')
        self.limit_tol = float(joint_limit_tolerance)
        if not np.isfinite(self.limit_tol) or self.limit_tol < 0:
            raise ValueError('joint_limit_tolerance must be nonnegative and finite')
        dynamic = {j.name: j for j in self.joints if j.kind != 'fixed'}
        if not set(self.names[0]).issubset(dynamic):
            raise ValueError(f'DOF map missing USD joints: {set(self.names[0])-set(dynamic)}')
        self.limits = [(dynamic[n].lower, dynamic[n].upper) for n in self.names[0]]
        self.prismatic = [dynamic[n].kind == 'prismatic' for n in self.names[0]]
        self.managers = [trimesh.collision.CollisionManager() for _ in range(2)]
        for manager in self.managers:
            for i, (_, _, vertices, faces, _) in enumerate(self.colliders):
                manager.add_object(str(i), trimesh.Trimesh(vertices=vertices, faces=faces, process=False))

    def _q(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (13,) or not np.isfinite(q).all():
            raise InvalidConfiguration('Each q must be a finite array of 13 DOFs')
        for i, (lo, hi) in enumerate(self.limits):
            if q[i] < lo-self.limit_tol or q[i] > hi+self.limit_tol:
                raise InvalidConfiguration(f'{self.names[0][i]} index={i} value={q[i]:.9g} outside [{lo:.9g}, {hi:.9g}] tolerance={self.limit_tol:g}')
        return q

    def check(self, q0, q1, base0, base1):
        """Raises on invalid states; returns collision-free status + robot0/robot1 pairs."""
        qs = [self._q(q0), self._q(q1)]
        bases = [_pose7(base0), _pose7(base1)]
        for r in range(2):
            poses = forward_graph(self.joints, ROOT, dict(zip(self.names[r], qs[r])))
            world_base = mat(bases[r][:3], bases[r][3:])
            for i, (_, link, _, _, local) in enumerate(self.colliders):
                if link not in poses:
                    raise RuntimeError(f'FK missing collision link: {link}')
                self.managers[r].set_transform(str(i), world_base @ poses[link] @ local)
        collided, pairs = self.managers[0].in_collision_other(self.managers[1], return_names=True)
        named = sorted([(self.colliders[int(i)][0], self.colliders[int(j)][0]) for i, j in pairs]) if collided else []
        return CollisionResult(collision_free=not bool(collided), pairs=named)

    def is_collision_free(self, q0, q1, base0, base1):
        return self.check(q0, q1, base0, base1).collision_free

    def check_edge(self, start, goal, max_revolute_step=0.05, max_prismatic_step=0.01,
                   max_base_translation_step=0.01, max_base_rotation_step=0.05,
                   max_samples=10001):
        """Validate a LINEAR JOINT + LINEAR BASE POSITION/SLERP ROTATION edge.

        start/goal each have keys q0, q1, base0, base1. Includes endpoints.
        No continuous guarantee between samples. Base poses are interpolated only if
        motion is allowed by the planning model; base movement feasibility is NOT checked.
        """
        steps = [max_revolute_step, max_prismatic_step,
                 max_base_translation_step, max_base_rotation_step]
        if any(not np.isfinite(s) or s <= 0 for s in steps):
            raise ValueError('All maximum steps must be positive finite numbers')
        if not isinstance(max_samples, int) or max_samples < 2:
            raise ValueError('max_samples must be integer >= 2')
        a = {k: self._q(start[k]) if k.startswith('q') else _pose7(start[k]) for k in ('q0','q1','base0','base1')}
        b = {k: self._q(goal[k]) if k.startswith('q') else _pose7(goal[k]) for k in ('q0','q1','base0','base1')}
        n = 1
        for key in ('q0','q1'):
            d = np.abs(b[key]-a[key])
            for i, dx in enumerate(d):
                s = max_prismatic_step if self.prismatic[i] else max_revolute_step
                n = max(n, int(math.ceil(dx / s)))
        for key in ('base0','base1'):
            n = max(n, int(math.ceil(np.linalg.norm(b[key][:3]-a[key][:3]) / max_base_translation_step)))
            n = max(n, int(math.ceil(quaternion_angle(a[key],b[key]) / max_base_rotation_step)))
        if n+1 > max_samples:
            raise RuntimeError(f'Edge requires {n+1} samples; max_samples={max_samples}; refusing unsafe undersampling')
        for i in range(n+1):
            t = i/n
            qs = {k: (1-t)*a[k]+t*b[k] for k in ('q0','q1')}
            bases = {k: interpolate_pose(a[k],b[k],t) for k in ('base0','base1')}
            result = self.check(qs['q0'],qs['q1'],bases['base0'],bases['base1'])
            if not result.collision_free:
                return {'collision_free':False, 'sample_index':i,'sample_count':n+1,
                        't':t, 'pairs':result.pairs}
        return {'collision_free':True, 'sample_index':None,
                'sample_count':n+1, 't':None,'pairs':[]}
