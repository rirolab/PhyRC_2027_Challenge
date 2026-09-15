"""Read-only contact criterion on authored collision surfaces, NOT solver events.

Contact means triangle-to-triangle distance <= the sum of the existing authored
contact offsets. Spheres use their analytic radius. PhysX SDF/convex cooking and
speculative contacts can differ from these surfaces. No physics is modified.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def point_triangle_distance_squared(p, tri):
    """Vectorized point/triangle distance, including edges and degenerate faces."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    u, v, w = b - a, c - a, p - a
    normal = np.cross(u, v)
    area2 = np.einsum('ij,ij->i', normal, normal)
    distance_plane = np.einsum('ij,ij->i', w, normal)
    projection = p - normal * (distance_plane / np.maximum(area2, 1e-300))[:, None]
    wu = np.einsum('ij,ij->i', projection - a, u)
    wv = np.einsum('ij,ij->i', projection - a, v)
    uu = np.einsum('ij,ij->i', u, u)
    uv = np.einsum('ij,ij->i', u, v)
    vv = np.einsum('ij,ij->i', v, v)
    denom = np.maximum(uu * vv - uv * uv, 1e-300)
    s, t = (wu * vv - wv * uv) / denom, (wv * uu - wu * uv) / denom
    inside = (area2 > 1e-28) & (s >= 0) & (t >= 0) & (s + t <= 1)
    result = np.where(inside, distance_plane ** 2 / np.maximum(area2, 1e-300), np.inf)
    for x, y in ((a, b), (b, c), (c, a)):
        edge = y - x
        fraction = np.clip(np.einsum('ij,ij->i', p - x, edge) /
                           np.maximum(np.einsum('ij,ij->i', edge, edge), 1e-300), 0, 1)
        delta = p - x - fraction[:, None] * edge
        result = np.minimum(result, np.einsum('ij,ij->i', delta, delta))
    return result


def segment_distance_squared(p, q, a, b):
    """Closest points of paired closed line segments (parallel/degenerate safe)."""
    u, v, w = q - p, b - a, p - a
    dot = lambda x, y: np.einsum('ij,ij->i', x, y)
    uu, uv, vv, uw, vw = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    denom = uu * vv - uv * uv
    s = np.clip(np.divide(uv * vw - vv * uw, denom,
                          out=np.zeros_like(denom), where=denom > 1e-28), 0, 1)
    t = (uv * s + vw) / np.maximum(vv, 1e-300)
    s = np.where(t < 0, np.clip(-uw / np.maximum(uu, 1e-300), 0, 1), s)
    s = np.where(t > 1, np.clip((uv - uw) / np.maximum(uu, 1e-300), 0, 1), s)
    t = np.clip(t, 0, 1)
    # If the second segment degenerates to a point, optimize the first alone.
    s = np.where(vv < 1e-28, np.clip(-uw / np.maximum(uu, 1e-300), 0, 1), s)
    delta = w + s[:, None] * u - t[:, None] * v
    return dot(delta, delta)


def segment_hits_triangle(p, q, tri):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    normal = np.cross(b - a, c - a)
    denom = np.einsum('ij,ij->i', q - p, normal)
    fraction = np.divide(np.einsum('ij,ij->i', a - p, normal), denom,
                         out=np.full_like(denom, -1), where=np.abs(denom) > 1e-20)
    point = p + fraction[:, None] * (q - p)
    return (fraction >= 0) & (fraction <= 1) & (point_triangle_distance_squared(point, tri) <= 1e-24)


def triangle_pairs_touch(first, second, margin):
    """Complete triangle surface test: vertices, edge/edge and edge/face hits."""
    limit = margin * margin
    for i in range(3):
        if np.any(point_triangle_distance_squared(first[:, i], second) <= limit):
            return True
        if np.any(point_triangle_distance_squared(second[:, i], first) <= limit):
            return True
        p, q = first[:, i], first[:, (i + 1) % 3]
        if np.any(segment_hits_triangle(p, q, second)):
            return True
        if np.any(segment_hits_triangle(second[:, i], second[:, (i + 1) % 3], first)):
            return True
        for j in range(3):
            if np.any(segment_distance_squared(p, q, second[:, j], second[:, (j + 1) % 3]) <= limit):
                return True
    return False


class TriangleTree:
    def __init__(self, triangles):
        self.triangles = triangles
        self.lower, self.upper = triangles.min(1), triangles.max(1)
        centres = (self.lower + self.upper) / 2
        def build(ids):
            lo, hi = self.lower[ids].min(0), self.upper[ids].max(0)
            if len(ids) <= 16:
                return (lo, hi, ids, None, None)
            axis = int(np.argmax(np.ptp(centres[ids], axis=0)))
            ids = ids[np.argsort(centres[ids, axis], kind='stable')]
            middle = len(ids) // 2
            return (lo, hi, None, build(ids[:middle]), build(ids[middle:]))
        self.root = build(np.arange(len(triangles)))

    def touches(self, triangles, lower, upper, margin):
        # Traverse one static BVH with vectorized moving-triangle candidate sets.
        pending = [(self.root, np.arange(len(triangles)))]
        while pending:
            node, ids = pending.pop()
            lo, hi, faces, left, right = node
            ids = ids[((upper[ids] >= lo - margin) & (lower[ids] <= hi + margin)).all(1)]
            if not len(ids):
                continue
            if faces is None:
                pending.extend(((left, ids), (right, ids)))
                continue
            for start in range(0, len(ids), 128):
                batch = ids[start:start + 128]
                overlap = ((upper[batch, None] >= self.lower[faces] - margin) &
                           (lower[batch, None] <= self.upper[faces] + margin)).all(2)
                moving, fixed = np.nonzero(overlap)
                if len(moving) and triangle_pairs_touch(triangles[batch[moving]], self.triangles[faces[fixed]], margin):
                    return True
        return False


class CollisionContact:
    def __init__(self, context, human_world):
        if context.get('version') != 1:
            raise ValueError('Unsupported collision contact context')
        self.context = context
        self.faces = np.asarray(context['garment_faces'], dtype=int)
        self.colliders = []
        matrix = np.asarray(human_world, dtype=float)
        for shape in context['colliders']:
            margin = context['garment_contact_offset_m'] + shape['contact_offset_m']
            if not np.isfinite(margin) or margin < 0:
                raise ValueError('Invalid authored contact offsets')
            points = np.asarray(shape['points'], dtype=float)
            world = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
            if shape['type'] == 'sphere':
                scale = np.linalg.norm(matrix[:3, :3], axis=1)
                if not np.allclose(scale, scale[0], rtol=1e-6):
                    raise ValueError('Nonuniform sphere scale is unsupported')
                self.colliders.append((shape['path'], 'sphere', (world[0], shape['radius'] * scale[0]), margin))
            else:
                tree = TriangleTree(world[np.asarray(shape['faces'], dtype=int)])
                self.colliders.append((shape['path'], 'mesh', tree, margin))
        if not self.colliders:
            raise ValueError('No enabled mannequin collision shapes')
        self.last_collider = None

    def touches(self, points):
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite cloth contact state')
        tri = np.asarray(points, dtype=float)[self.faces]
        lower, upper = tri.min(1), tri.max(1)
        for path, kind, shape, margin in self.colliders:
            if kind == 'sphere':
                centre, radius = shape
                candidate = ((upper >= centre - radius - margin) & (lower <= centre + radius + margin)).all(1)
                contact = bool(np.any(point_triangle_distance_squared(centre, tri[candidate]) <= (radius + margin) ** 2))
            else:
                contact = shape.touches(tri, lower, upper, margin)
            if contact:
                self.last_collider = path
                return True
        return False


def triangulate(mesh):
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    faces, offset = [], 0
    for count in counts:
        face = indices[offset:offset + count]
        faces.extend([[face[0], face[i], face[i + 1]] for i in range(1, count - 1)])
        offset += count
    return np.asarray(faces, dtype=int)


def contact_context_from_stage(stage, garment_path):
    """Read authored shapes and offsets; never Apply/Set a USD physics property."""
    from pxr import Usd, UsdGeom, UsdPhysics
    root = stage.GetPrimAtPath('/World/Human')
    root_matrix = np.asarray(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    inverse = np.linalg.inv(root_matrix)
    garment = stage.GetPrimAtPath(garment_path)
    def contact_offset(prim):
        value = prim.GetAttribute('physxCollision:contactOffset').Get()
        if value is None or not np.isfinite(value) or value < 0:
            raise ValueError(f'Explicit finite contactOffset required: {prim.GetPath()}')
        return float(value)
    colliders = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue  # CollisionAPI on the SkelRoot is not a geometric shape.
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())) @ inverse
        shape = {'path': str(prim.GetPath()), 'contact_offset_m': contact_offset(prim),
                 'approximation': prim.GetAttribute('physics:approximation').Get()}
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            shape.update(type='mesh', faces=triangulate(mesh).tolist())
        elif prim.IsA(UsdGeom.Sphere):
            points = np.zeros((1, 3))
            scale = np.linalg.norm(matrix[:3, :3], axis=1)
            if not np.allclose(scale, scale[0], rtol=1e-6):
                raise ValueError('Nonuniform sphere scale is unsupported')
            shape.update(type='sphere', radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()) * scale[0])
        else:
            raise ValueError(f'Unsupported mannequin collider: {prim.GetPath()} ({prim.GetTypeName()})')
        shape['points'] = ((np.c_[points, np.ones(len(points))] @ matrix)[:, :3]).tolist()
        colliders.append(shape)
    if not garment.IsA(UsdGeom.Mesh) or not colliders:
        raise ValueError('Garment mesh and enabled mannequin colliders are required')
    return {'version': 1, 'criterion': 'authored_collision_surface_distance_with_contact_offsets',
            'native_physx_contact_event': False,
            'limitations': 'Authored SDF/hull source mesh, not cooked PhysX surfaces or speculative solver contacts; sampled physics states only.',
            'garment_path': garment_path, 'garment_contact_offset_m': contact_offset(garment),
            'garment_faces': triangulate(UsdGeom.Mesh(garment)).tolist(), 'colliders': colliders}


def recording_contact_context(reader, context_path=None):
    """New archives are self-contained. Older v2 can use a checksum-bound sidecar."""
    header, initial = next(reader.frames())
    context = json.loads(str(initial['evaluation_context'])) if 'evaluation_context' in initial else {}
    if 'contact' in context:
        return None
    path = Path(context_path) if context_path else reader.path.parent.parent / 'evaluation/contact_contexts' / (reader.path.name + '.json')
    if not path.is_file():
        raise ValueError('This recording needs a one-time collision context export (no GPU): '
                         f'PHYRC_ACCEPT_EULA=1 ./run.sh cpu /scripts/export_replay_contact.py '
                         f'/output/full_teleop/{reader.path.name} . Then run evaluation again.')
    data = json.loads(path.read_text())
    if data['scene_sha256'] != reader.manifest['scene_sha256'] or data['manifest_sha256'] != hashlib.sha256((reader.path / 'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Collision context does not belong to this recording')
    if data['scene_sha256'] != hashlib.sha256((reader.path / 'scene.usdc').read_bytes()).hexdigest():
        raise ValueError('Recorded scene checksum mismatch')
    return data['contact']
