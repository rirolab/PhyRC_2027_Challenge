"""Posed hand hulls and legacy wrist spheres for Isaac Sim 6 surface cloth.

Adapted from vendor/hand_sphere_collider_handoff. Input hand points are already
in the human root's frame; applying the source mesh transform again is wrong.
PhysX uses analytic spheres. The existing triangle sweep guard uses tightly
circumscribed sphere meshes, including for checkpoint validation.
"""
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt


def _unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1.e-10:
        raise ValueError("cannot derive a hand collider direction")
    return np.asarray(vector) / length


def _uniform_scale(transform):
    basis = np.asarray([transform.TransformDir(Gf.Vec3d(*row)) for row in np.eye(3)])
    scale = float(np.linalg.norm(basis[0]))
    if scale <= 0 or not np.allclose(basis @ basis.T, np.eye(3) * scale**2,
                                     rtol=1.e-5, atol=1.e-8):
        raise ValueError("hand spheres require a uniform human root scale without shear")
    return scale


def _cut_seam_fit(stage, root_path, wrist_centre, root_xf, scale):
    """Smallest sphere enclosing the actual skinned forearm cut boundary."""
    from scipy.optimize import minimize
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    body = stage.GetPrimAtPath(root_path + '/CollisionBody')
    if not body or not body.IsA(UsdGeom.Mesh):
        return None
    mesh = UsdGeom.Mesh(body)
    if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
        raise ValueError('wrist seam fitting requires a triangular body collider')
    tri = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    edges, counts = np.unique(np.sort(np.concatenate(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1),
        axis=0, return_counts=True)
    boundary = edges[counts == 1]
    if not len(boundary):
        return None  # Explicit wrist_sphere mode retains the complete hand.
    ids, inverse = np.unique(boundary, return_inverse=True)
    pairs = inverse.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(ids), len(ids)))
    count, labels = connected_components(graph, directed=False)
    xf = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * root_xf.GetInverse()
    points = np.asarray([xf.Transform(Gf.Vec3d(*map(float, p)))
                         for p in np.asarray(mesh.GetPointsAttr().Get())[ids]])
    rings = [points[labels == k] for k in range(count)]
    ring = min(rings, key=lambda p: np.linalg.norm(p.mean(0) - wrist_centre))
    if np.linalg.norm(ring.mean(0) - wrist_centre) * scale > .12:
        return None  # Do not mistake a neck boundary for a missing wrist cut.
    origin = ring.mean(0)
    unit = np.linalg.norm(ring - origin, axis=1).max()
    q = (ring - origin) / unit
    fit = minimize(lambda x: x[3], np.array([0., 0., 0., 1.]),
                   method='SLSQP', bounds=[(None, None)] * 3 + [(0, None)],
                   constraints={'type': 'ineq', 'fun': lambda x:
                                x[3] ** 2 - ((q - x[:3]) ** 2).sum(1)},
                   options={'ftol': 1.e-12, 'maxiter': 100})
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('could not fit a continuous wrist/sphere seam')
    return origin + unit * fit.x[:3], ring


def fit_wrist_spheres(stage, root_path, hand_data, _feel, radius_m=0.0):
    """Fit posed cut seams, using radius_m as the minimum collision radius.

    HUMAN_HAND_SPHERE_SEAM_OVERLAP=0 restores the earlier anatomical fit and
    its sink/lateral offsets. Positive overlap uses the actual CollisionBody
    boundary; UP still shifts the centre and radius grows to retain coverage.
    """
    mesh, points, _, _, hands, groups = hand_data
    if mesh is None or set(hands) != {"left", "right"}:
        raise ValueError("wrist spheres require both skinned hand vertex sets")
    root = stage.GetPrimAtPath(root_path)
    root_xf = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    scale = _uniform_scale(root_xf)
    mesh_xf = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mesh_to_root = mesh_xf * root_xf.GetInverse()
    # The SMPL-X raw X direction survives a nested rotation of the visual mesh.
    lateral = _unit(mesh_to_root.TransformDir(Gf.Vec3d(1, 0, 0)))
    origin = np.asarray(mesh_to_root.Transform(Gf.Vec3d(0, 0, 0)))
    # UP means world +Z in this Z-up stage, regardless of the asset's raw axes.
    up = _unit(root_xf.GetInverse().TransformDir(Gf.Vec3d(0, 0, 1)))
    fit_pct = _feel("HUMAN_COLLIDER_FIT_PCT", 92.0)
    sink = _feel("HUMAN_HAND_SPHERE_SINK", 2.6) + _feel("HUMAN_HAND_SPHERE_SINK_EXTRA", 0.35)
    outward = _feel("HUMAN_HAND_SPHERE_OUTWARD", 0.03)
    upward = _feel("HUMAN_HAND_SPHERE_UP", 0.0)
    medial = _feel("HUMAN_HAND_SPHERE_MEDIAL", 0.02)
    seam_overlap = _feel("HUMAN_HAND_SPHERE_SEAM_OVERLAP", 0.002)
    side_lateral = {"left": _feel("HUMAN_HAND_SPHERE_LEFT_LATERAL", 0.01),
                    "right": _feel("HUMAN_HAND_SPHERE_RIGHT_LATERAL", 0.0)}
    params = [fit_pct, sink, outward, upward, medial, radius_m, seam_overlap, *side_lateral.values()]
    if not np.isfinite(params).all() or not 0 < fit_pct <= 100 or radius_m < 0 or seam_overlap < 0:
        raise ValueError("invalid hand sphere parameters (fit percentile must be in (0, 100])")
    result = []
    for side, indices in hands.items():
        hp = points[indices]
        centre = hp.mean(axis=0)
        axis = np.linalg.svd(hp - centre, full_matrices=False)[2][0]
        finger_groups = [groups[(side, f)] for f in ("index", "middle", "ring", "pinky")
                         if (side, f) in groups]
        if not finger_groups:
            raise ValueError(f"no finger weights for {side} hand")
        fingers = np.concatenate(finger_groups)
        if np.median((points[fingers] - centre) @ axis) < 0:
            axis = -axis
        all_fingers = np.concatenate(finger_groups +
                                     ([groups[(side, "thumb")]] if (side, "thumb") in groups else []))
        wrist_ids = np.setdiff1d(indices, all_fingers)
        if len(wrist_ids) < 3:
            raise ValueError(f"insufficient wrist weights for {side} hand")
        wrist = points[wrist_ids]
        wrist_centre = wrist.mean(axis=0)
        rel = wrist - wrist_centre
        perpendicular = rel - np.outer(rel @ axis, axis)
        fitted_radius = float(np.percentile(np.linalg.norm(perpendicular, axis=1), fit_pct))
        radius = radius_m / scale if radius_m > 0 else fitted_radius
        if not np.isfinite(radius) or radius < 1.e-6:
            raise ValueError(f"degenerate {side} wrist radius")
        toward_midline = -1.0 if np.dot(wrist_centre - origin, lateral) > 0 else 1.0
        # The handoff's tuned centre uses the anatomical wrist fit. Changing
        # only the collision radius must not also move the sphere toward the
        # fingers and open a gap at the cut forearm. Keep that reference for
        # sink; all handed offsets retain their original values and axes.
        position = (wrist_centre + axis * (fitted_radius * (1.0 - sink) + outward / scale)
                    + up * (upward / scale)
                    + lateral * toward_midline * (medial - side_lateral[side]) / scale)
        seam_fitted = False
        if seam_overlap > 0:
            seam = _cut_seam_fit(stage, root_path, wrist_centre, root_xf, scale)
            if seam is not None:
                # Geometric seam fitting replaces the old anatomical sink and
                # lateral offsets. Zero overlap restores that legacy placement.
                # Requested radius is a minimum; every cut edge must stay inside
                # the sphere, with overlap, for both native contact and the guard.
                position, ring = seam
                seam_fitted = True
                position = position + up * (upward / scale)
                radius = max(radius, np.linalg.norm(ring - position, axis=1).max()
                             + seam_overlap / scale)
        result.append(dict(side=side, centre=position, radius=radius,
                           seam_fitted=seam_fitted,
                           radius_m=radius * scale, fitted_radius_m=fitted_radius * scale, hand_vertices=len(indices),
                           wrist_vertices=len(wrist_ids),
                           hand_length_m=float(np.ptp((hp - centre) @ axis) * scale)))
    return result


def build_hand_spheres(stage, root_path, hand_data, _feel, radius_m=0.0):
    result = fit_wrist_spheres(stage, root_path, hand_data, _feel, radius_m)
    out = []
    for item in result:
        path = f"{root_path}/HandSphere_{item['side']}"
        sphere = UsdGeom.Sphere.Define(stage, path)
        radius = item['radius']
        sphere.CreateRadiusAttr(radius)
        sphere.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-radius), Gf.Vec3f(radius)]))
        xf = UsdGeom.Xformable(sphere)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, item['centre'])))
        UsdGeom.Imageable(sphere).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim()).CreateCollisionEnabledAttr(True)
        sphere.GetPrim().SetCustomDataByKey("handSphereFit",
            "cut seam enclosure; human-root coordinates" if item['seam_fitted']
            else "wrist percentile; human-root coordinates")
        out.append((path, item['radius_m'], item['hand_length_m'], item['hand_vertices']))
        print(f"[Teleop] wrist sphere {item['side']}: radius={item['radius_m']:.5f}m, "
              f"{item['wrist_vertices']} wrist vertices, root centre={item['centre'].round(5)}",
              flush=True)
    return out


def build_fitted_hand_colliders(stage, root_path, hand_data, _feel):
    """Close finger gaps while matching the complete visible hand and wrist cut.

    Both native convex contact and the triangle guard consume the same closed
    hull. Neither arm length nor the visual mesh is changed. A small uniform
    expansion keeps every original point inside, with <= margin displacement.
    """
    from scipy.spatial import ConvexHull
    from pxr import PhysxSchema
    mesh, points, _, _, hands, groups = hand_data
    if mesh is None or set(hands) != {"left", "right"}:
        raise ValueError("fitted hand colliders require both posed hand vertex sets")
    root_xf = UsdGeom.Xformable(stage.GetPrimAtPath(root_path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    scale = _uniform_scale(root_xf)
    margin = float(_feel("HUMAN_HAND_HULL_MARGIN", 0.001))
    overlap = float(_feel("HUMAN_HAND_HULL_SEAM_OVERLAP", 0.002))
    if not np.isfinite([margin, overlap]).all() or margin < 0 or overlap <= 0:
        raise ValueError("hand hull margin must be nonnegative and seam overlap positive")
    result = []
    for side, indices in hands.items():
        hp = points[indices]
        finger_sets = [v for (s, _), v in groups.items() if s == side]
        fingers = np.concatenate(finger_sets) if finger_sets else np.array([], dtype=int)
        wrist_ids = np.setdiff1d(indices, fingers)
        if len(wrist_ids) < 3:
            raise ValueError(f"insufficient wrist vertices for {side} hand hull")
        wrist = points[wrist_ids].mean(0)
        seam = _cut_seam_fit(stage, root_path, wrist, root_xf, scale)
        if seam is None:
            raise ValueError(f"missing forearm cut boundary for {side} hand hull")
        _, ring = seam
        toward_arm = _unit(ring.mean(0) - hp.mean(0))
        cloud = np.concatenate((hp, ring, ring + toward_arm * (overlap / scale)))
        initial = ConvexHull(cloud)
        vertices = cloud[initial.vertices]
        centre = vertices.mean(0)
        extent = float(np.linalg.norm(vertices - centre, axis=1).max())
        vertices = centre + (vertices - centre) * (1.0 + margin / (scale * extent))
        hull = ConvexHull(vertices)
        if len(vertices) > 256:
            raise ValueError(f"{side} hand hull exceeds the native 256-vertex limit")
        triangles = hull.simplices.copy()
        normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                           vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
        reverse = np.einsum('ij,ij->i', normals, vertices[triangles[:, 0]] - centre) < 0
        triangles[reverse] = triangles[reverse, ::-1]
        path = f"{root_path}/HandHull_{side}"
        shape = UsdGeom.Mesh.Define(stage, path)
        shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        shape.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32)))
        shape.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.astype(np.int32).ravel()))
        shape.CreateExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(shape.GetPointsAttr().Get()))
        shape.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdGeom.Imageable(shape).MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
        PhysxSchema.PhysxConvexHullCollisionAPI.Apply(shape.GetPrim()).CreateHullVertexLimitAttr(256)
        shape.GetPrim().SetCustomDataByKey('handColliderFit', 'posed visual hand plus forearm seam')
        shape.GetPrim().SetCustomDataByKey('handColliderMarginM', margin)
        result.append(path)
        print(f"[Teleop] fitted hand {side}: {len(indices)} visual vertices, "
              f"hull={len(vertices)} vertices/{len(triangles)} faces, "
              f"max expansion={margin:.4f}m, seam overlap={overlap:.4f}m", flush=True)
    return result


def sphere_world_geometry(prim):
    sphere = UsdGeom.Sphere(prim)
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return (np.asarray(xf.Transform(Gf.Vec3d(0, 0, 0))),
            float(sphere.GetRadiusAttr().Get()) * _uniform_scale(xf))


def _sphere_guard_mesh():
    """Closed outward icosphere, circumscribed to avoid gaps inside the sphere."""
    from scipy.spatial import ConvexHull
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([(0, a, b * phi) for a in (-1, 1) for b in (-1, 1)] +
                        [(a, b * phi, 0) for a in (-1, 1) for b in (-1, 1)] +
                        [(a * phi, 0, b) for a in (-1, 1) for b in (-1, 1)], dtype=float)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    for _ in range(3):
        faces = ConvexHull(vertices).simplices
        edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                                   faces[:, [2, 0]])), axis=1), axis=0)
        midpoints = vertices[edges].mean(axis=1)
        midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)
        vertices = np.concatenate((vertices, midpoints))
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                     vertices[faces[:, 2]] - vertices[faces[:, 0]])
    reverse = np.einsum('ij,ij->i', cross, vertices[faces[:, 0]]) < 0
    faces[reverse] = faces[reverse, ::-1]
    vertices /= np.min(-hull.equations[:, 3])
    return vertices, faces


def append_hand_sphere_surfaces(body_prim, points, triangles):
    """Include the actual enabled hand spheres or hulls in the surface guard."""
    parent = body_prim.GetParent()
    # Visible meshes can be nested; locate the nearest ancestor owning spheres.
    spheres = []
    while parent and not parent.IsPseudoRoot():
        spheres = [p for p in parent.GetChildren()
                   if ((p.IsA(UsdGeom.Sphere) and p.GetName().startswith('HandSphere_'))
                       or (p.IsA(UsdGeom.Mesh) and p.GetName().startswith('HandHull_')))
                   and p.HasAPI(UsdPhysics.CollisionAPI)
                   and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
        if spheres:
            break
        parent = parent.GetParent()
    if not spheres:
        return points, triangles
    unit_points, unit_tris = (None, None)
    for prim in spheres:
        if prim.IsA(UsdGeom.Sphere):
            if unit_points is None:
                unit_points, unit_tris = _sphere_guard_mesh()
            centre, radius = sphere_world_geometry(prim)
            vertices, faces = centre + radius * unit_points, unit_tris
        else:
            mesh = UsdGeom.Mesh(prim)
            if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
                raise ValueError('fitted hand guard requires triangular hull faces')
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            vertices = np.asarray([xf.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()])
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        triangles = np.concatenate((triangles, faces + len(points)))
        points = np.concatenate((points, vertices))
    print(f"[Teleop] surface guard includes {len(spheres)} enabled hand collision shapes", flush=True)
    return points, triangles
