"""Numerical dressing measurements; no simulation or physics mutations.

V7 awards binary upper-arm enclosure by the routed sleeve material only.
Full-garment coverage remains a legacy diagnostic. Virtual caps and centreline
containment are read-only geometric proxies; they do not mutate the garment.
"""
import numpy as np


def vneck_front_vertices(mesh):
    """Read authored V-neck material-region identity, never infer front from pose."""
    from pxr import UsdGeom
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3)
    subset = UsdGeom.Subset(mesh.GetPrim().GetChild('vneck'))
    if not subset or subset.GetElementTypeAttr().Get() != 'face':
        raise ValueError('Overall dressing requires an authored vneck face subset')
    ids = np.asarray(subset.GetIndicesAttr().Get(), int)
    if not len(ids) or ids.min() < 0 or ids.max() >= len(faces):
        raise ValueError('Invalid vneck face subset')
    return np.unique(faces[ids])


def front_alignment(points, collar, front_ids, arms, neck_chain):
    """Signed shirt-front/body-front alignment, invariant to mannequin spawn yaw."""
    up = np.asarray(neck_chain[-1]) - neck_chain[0]
    up /= np.linalg.norm(up)
    forward = np.cross(np.asarray(arms['left'][0]) - arms['right'][0], up)
    length = np.linalg.norm(forward)
    if length < 1e-6:
        raise ValueError('Cannot determine anatomical forward from shoulders and spine')
    forward /= length
    centre = points[collar].mean(0)
    delta = points[front_ids].mean(0) - centre
    delta -= np.dot(delta, up) * up
    separation = float(np.linalg.norm(delta))
    cosine = float(np.dot(delta, forward) / separation) if separation > 1e-6 else 0.0
    # A collapsed/folded collar with no identifiable front separation earns none.
    correct = separation >= .02 and cosine >= np.cos(np.deg2rad(45))
    return {'front_facing': bool(correct), 'alignment_cosine': cosine,
            'front_separation_m': separation, 'max_angle_deg': 45.0,
            'min_front_separation_m': .02, 'body_forward_world': forward.tolist()}


def boundary_loops(faces):
    faces = np.asarray(faces, dtype=int)
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0,
                                    return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Garment must be a manifold triangle mesh')
    edges = directed[counts[inverse] == 1]
    following = dict(map(tuple, edges))
    if len(following) != len(edges) or set(following) != set(following.values()):
        raise ValueError('Garment boundary must have consistently oriented simple loops')
    loops = []
    while following:
        start = next(iter(following))
        loop, current = [], start
        while current in following:
            loop.append(current)
            current = following.pop(current)
        if current != start or len(loop) < 3:
            raise ValueError('Broken garment boundary')
        loops.append(np.asarray(loop, dtype=int))
    return loops


def capped_mesh(points, faces, loops):
    """Close openings for numeric containment ONLY; never write these to USD."""
    vertices = np.vstack((points, [points[ids].mean(0) for ids in loops]))
    caps = [[int(b), int(a), len(points) + i] for i, ids in enumerate(loops)
            for a, b in zip(ids, np.roll(ids, -1))]
    return vertices, np.vstack((faces, np.asarray(caps)))


def inside_mesh(queries, points, faces):
    """Generalized winding magnitude > 1/2 on an oriented closed surface."""
    queries, points = np.asarray(queries, float), np.asarray(points, float)
    result = np.zeros(len(queries), bool)
    if not len(queries):
        return result
    candidate = np.all((queries > points.min(0)) & (queries < points.max(0)), axis=1)
    triangles = points[np.asarray(faces)]
    for index in np.flatnonzero(candidate):
        a, b, c = (triangles[:, i] - queries[index] for i in range(3))
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum('ij,ij->i', a, np.cross(b, c))
        denominator = (la * lb * lc + np.einsum('ij,ij->i', a, b) * lc
                       + np.einsum('ij,ij->i', b, c) * la
                       + np.einsum('ij,ij->i', c, a) * lb)
        winding = np.sum(2 * np.arctan2(numerator, denominator)) / (4 * np.pi)
        result[index] = abs(winding) > 0.5
    return result


def cuff_crossing(loop, inner_points, chain, planarity_limit=0.6):
    """Find an arm-chain crossing of a cuff polygon in either direction.

    chain contains shoulder, elbow, wrist, and a distal hand point. Nonplanar
    or collapsed cuffs are rejected instead of inferring entry from proximity.
    Returns the crossing position along the polyline, or None.
    """
    centre = loop.mean(0)
    _, singular, axes = np.linalg.svd(loop - centre, full_matrices=False)
    if singular[1] < 1e-5 or singular[2] / singular[1] > planarity_limit:
        return None
    normal = axes[2]
    if np.dot(normal, centre - inner_points.mean(0)) < 0:
        normal = -normal
    polygon = (loop - centre) @ axes[:2].T
    distance = 0.0
    for p, q in zip(chain, chain[1:]):
        length = np.linalg.norm(q - p)
        a, b = np.dot(p - centre, normal), np.dot(q - centre, normal)
        if (a < -1e-5 and b > 1e-5) or (a > 1e-5 and b < -1e-5):
            fraction = -a / (b - a)
            point = (p + fraction * (q - p) - centre) @ axes[:2].T
            inside = False
            for u, v in zip(polygon, np.roll(polygon, -1, axis=0)):
                if (u[1] > point[1]) != (v[1] > point[1]):
                    x = u[0] + (point[1] - u[1]) * (v[0] - u[0]) / (v[1] - u[1])
                    if point[0] < x:
                        inside = not inside
            if inside:
                return float(distance + fraction * length)
        distance += length
    return None


def sleeve_regions(rest_points, faces, cuffs, collar, loops):
    """Freeze sleeve material triangles beyond the rest-shirt torso side seams.

    The current asset is authored flat with sleeve span along X. The hem gives
    the torso boundary; classification never follows the deformed world pose.
    Only whole triangles beyond that boundary are included (conservative at
    the seam). Each region must have exactly its cuff and one root opening.
    """
    rest_points, faces = np.asarray(rest_points), np.asarray(faces)
    hem = next(ids for ids in loops if not any(np.array_equal(ids, opening)
               for opening in [*cuffs, collar]))
    centre_x = float(rest_points[hem, 0].mean())
    regions = []
    for cuff in cuffs:
        sign = np.sign(rest_points[cuff, 0].mean() - centre_x)
        x = sign * (rest_points[:, 0] - centre_x)
        cut = float(x[hem].max())
        selected = np.flatnonzero(np.all(x[faces] > cut, axis=1))
        if not len(selected):
            raise ValueError('No sleeve triangles beyond the torso seam')
        # Curved torso side walls may protrude beyond the hem. Keep only the
        # connected component touching this cuff, excluding torso islands.
        adjacent = {}
        for index in selected:
            for vertex in faces[index]:
                adjacent.setdefault(int(vertex), []).append(int(index))
        pending = [index for vertex in cuff for index in adjacent.get(int(vertex), [])]
        visited, vertices_seen = set(), set()
        while pending:
            index = pending.pop()
            if index in visited:
                continue
            visited.add(index)
            for vertex in faces[index]:
                if int(vertex) not in vertices_seen:
                    vertices_seen.add(int(vertex))
                    pending.extend(adjacent[int(vertex)])
        selected = np.asarray(sorted(visited), int)
        if not len(selected):
            raise ValueError('Sleeve component does not reach its cuff')
        boundaries = boundary_loops(faces[selected])
        if len(boundaries) != 2 or not any(set(ids) == set(cuff) for ids in boundaries):
            raise ValueError('Sleeve annotation must contain one cuff and one root opening')
        regions.append(selected.tolist())
    return regions


def segment_in_mesh(segment, vertices, faces):
    """Any positive-length segment inside the volume; no coverage percentage.

    Split at triangle intersections and test interval midpoints. This avoids
    missing a short overlap between fixed coverage probes. 1e-8 m is solely
    numerical tolerance; surface tangency alone is not enclosure.
    """
    p, q = np.asarray(segment, float)
    direction = q - p
    triangles = vertices[faces]
    a = triangles[:, 0]
    e1, e2 = triangles[:, 1] - a, triangles[:, 2] - a
    h = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, h)
    valid = np.abs(det) > 1e-12
    inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
    delta = p - a
    u = np.einsum('ij,ij->i', delta, h) * inv
    cross = np.cross(delta, e1)
    v = cross @ direction * inv
    t = np.einsum('ij,ij->i', e2, cross) * inv
    valid &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0) & (t < 1)
    bounds = np.unique(np.r_[0., t[valid], 1.])
    widths = np.diff(bounds) * np.linalg.norm(direction)
    mids = (bounds[:-1] + bounds[1:]) * .5
    return bool(inside_mesh(p + mids[widths > 1e-8, None] * direction, vertices, faces).any())


class GarmentGeometry:
    def __init__(self, rest_points, faces, coverage_samples=40, front_ids=None):
        self.faces = np.asarray(faces, int)
        self.loops = boundary_loops(self.faces)
        if len(self.loops) != 4:
            raise ValueError('Evaluation currently requires a shirt with four openings')
        # Current T-shirt rest coordinates: sleeves are the two X extremes.
        ordered = sorted(self.loops, key=lambda ids: rest_points[ids, 0].mean())
        self.cuffs = [ordered[0], ordered[-1]]
        centres = [rest_points[ids, 0].mean() for ids in ordered]
        if min(centres[1] - centres[0], centres[-1] - centres[-2]) < 0.05:
            raise ValueError('Cannot identify two distinct rest-shape cuffs')
        self.inner = [np.setdiff1d(np.unique(self.faces[np.isin(self.faces, ids).any(1)]), ids)
                      for ids in self.cuffs]
        # Of the remaining torso openings the collar has the smaller perimeter.
        torso = ordered[1:-1]
        self.collar = min(torso, key=lambda ids: np.linalg.norm(
            np.diff(np.vstack((rest_points[ids], rest_points[ids[:1]])), axis=0), axis=1).sum())
        self.collar_inner = np.setdiff1d(np.unique(self.faces[np.isin(self.faces, self.collar).any(1)]), self.collar)
        self.coverage_samples = coverage_samples
        self.front_ids = None if front_ids is None else np.asarray(front_ids, int)
        self.set_sleeve_regions(sleeve_regions(rest_points, self.faces, self.cuffs, self.collar, self.loops))

    def set_sleeve_regions(self, regions):
        if len(regions) != len(self.cuffs):
            raise ValueError('Expected one sleeve region per cuff')
        self.sleeve_regions = [np.asarray(ids, int) for ids in regions]
        self.sleeve_loops = []
        for ids, cuff in zip(self.sleeve_regions, self.cuffs):
            if not len(ids) or ids.min() < 0 or ids.max() >= len(self.faces):
                raise ValueError('Invalid sleeve face indices')
            loops = boundary_loops(self.faces[ids])
            if len(loops) != 2 or not any(set(loop) == set(cuff) for loop in loops):
                raise ValueError('Sleeve region must match its cuff and one root')
            self.sleeve_loops.append(loops)

    def measure(self, points, arms, neck_chain=None, head_surface=None):
        vertices, closed_faces = capped_mesh(points, self.faces, self.loops)
        wrists, coverage, diagnostic = {}, {}, {}
        used_cuffs = {}
        for side, chain in arms.items():
            chain = np.asarray(chain)
            lengths = np.linalg.norm(np.diff(chain[:3], axis=0), axis=1)
            arm_length = lengths.sum()
            positions = (np.arange(self.coverage_samples) + 0.5) / self.coverage_samples * arm_length
            probes = np.array([chain[0] + d / lengths[0] * (chain[1] - chain[0])
                               if d < lengths[0] else
                               chain[1] + (d - lengths[0]) / lengths[1] * (chain[2] - chain[1])
                               for d in positions])
            crossings = [(i, cuff_crossing(points[ids], points[inner], chain))
                         for i, (ids, inner) in enumerate(zip(self.cuffs, self.inner))]
            valid = [(i, d) for i, d in crossings if d is not None]
            # A forearm/wrist must route through exactly one physical cuff.
            route = len(valid) == 1
            contained = inside_mesh(np.vstack((probes, chain[2:])), vertices, closed_faces)
            upper_probes = chain[0] + ((np.arange(self.coverage_samples) + .5)
                                      / self.coverage_samples)[:, None] * (chain[1] - chain[0])
            upper_coverage = float(inside_mesh(upper_probes, vertices, closed_faces).mean())
            i, distance = valid[0] if route else (None, None)
            # During first entry, wrist is inside and the cuff is still distal
            # to it. After advancement the wrist is outside past the cuff;
            # require some proximal arm containment to keep routing valid.
            proximal_inside = bool(contained[:-2].any())
            if (getattr(self, 'sleeve_regions', None) is not None and route
                    and distance < arm_length and not proximal_inside and not contained[-2]):
                proximal_inside = (segment_in_mesh(chain[:2], vertices, closed_faces)
                                   or segment_in_mesh(chain[1:3], vertices, closed_faces))
            wrists[side] = bool(route and (contained[-2] or
                                          (distance < arm_length and proximal_inside)))
            coverage[side] = float(contained[:-2].mean()) if wrists[side] else 0.0
            used_cuffs[side] = i if wrists[side] else None
            diagnostic[side] = {'cuff_index': i, 'crossing_distance_from_shoulder_m': distance,
                                'arm_length_m': float(arm_length), 'centreline_coverage': coverage[side],
                                'upper_arm_coverage': upper_coverage if wrists[side] else 0.0,
                                'hand_out': bool(wrists[side] and distance < arm_length
                                                 and not contained[-2:].any())}
            if getattr(self, 'sleeve_regions', None) is not None:
                covered = False
                if wrists[side]:
                    sleeve_vertices, sleeve_faces = capped_mesh(
                        points, self.faces[self.sleeve_regions[i]], self.sleeve_loops[i])
                    covered = segment_in_mesh(chain[:2], sleeve_vertices, sleeve_faces)
                diagnostic[side]['upper_arm_sleeve_covered'] = covered
        # Both arms passing the same opening do not dress two sleeves.
        if used_cuffs.get('left') is not None and used_cuffs.get('left') == used_cuffs.get('right'):
            wrists = {side: False for side in arms}
            coverage = {side: 0.0 for side in arms}
            for side in arms:
                diagnostic[side]['hand_out'] = False
                diagnostic[side]['upper_arm_coverage'] = 0.0
                if 'upper_arm_sleeve_covered' in diagnostic[side]:
                    diagnostic[side]['upper_arm_sleeve_covered'] = False
        if neck_chain is not None:
            # Chest must be in the torso, neck/head outside through the COLLAR,
            # not merely near the neckline or through a sleeve/hem.
            chain = np.asarray(neck_chain)
            crossing = cuff_crossing(points[self.collar], points[self.collar_inner], chain)
            inside = inside_mesh(chain, vertices, closed_faces)
            # A skeleton's neck joint can lie BELOW a properly seated collar.
            # Test exposed head skin instead of requiring that internal joint
            # to be outside. This rejects a neckline still caught on the face.
            neck_distance = float(np.linalg.norm(chain[1] - chain[0]))
            head_clearance = None
            if head_surface is not None:
                loop = points[self.collar]
                centre = loop.mean(0)
                _, _, axes = np.linalg.svd(loop - centre, full_matrices=False)
                normal = axes[2]
                # Head clearance needs the HEAD-facing half-space. The narrow
                # adjacent cloth ring can fold through the collar plane and
                # reverse its sign despite an almost stationary neckline.
                # Anatomical orientation is stateless and replay-independent.
                if np.dot(normal, chain[-1] - chain[0]) < 0:
                    normal = -normal
                head_clearance = float(np.min((np.asarray(head_surface) - centre) @ normal))
            neck_out = bool(crossing is not None and inside[0] and not inside[-1]
                            and (head_clearance >= -0.005 if head_clearance is not None
                                 else crossing < neck_distance and not inside[1]))
            diagnostic['neck'] = {'neck_out': neck_out, 'collar_crossing_from_chest_m': crossing,
                                  'chest_neck_head_inside': inside.tolist(),
                                  'clearance_normal_basis': 'anatomical_chest_to_head',
                                  'head_min_clearance_above_collar_m': head_clearance}
            diagnostic['dressing_complete'] = bool(neck_out and all(wrists.get(s, False)
                and diagnostic[s]['hand_out'] for s in ('left', 'right')))
            if getattr(self, 'front_ids', None) is not None:
                orientation = front_alignment(points, self.collar, self.front_ids, arms, chain)
                orientation['eligible_after_neck_out'] = neck_out
                diagnostic['orientation'] = orientation
                diagnostic['sleeves_and_neck_complete'] = diagnostic['dressing_complete']
                diagnostic['dressing_complete'] &= orientation['front_facing']
        return wrists, coverage, diagnostic
