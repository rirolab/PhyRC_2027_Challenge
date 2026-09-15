"""Trim the posterior skull in the posed SMPL-X mesh before collision copying."""
import json
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def round_posterior_head(points, head_weight):
    """Raw SMPL-X coordinates: X sideways, Y up, +Z toward the face."""
    p = np.asarray(points, dtype=float)
    weights = np.asarray(head_weight, dtype=float)
    selected = weights > 0.5
    if selected.sum() < 100:
        raise ValueError('Too few weighted head vertices for skull fitting')
    head = p[selected]
    crown = float(head[:, 1].max())
    height = float(np.ptp(head[:, 1]))
    upper = head[head[:, 1] > crown - 0.5 * height]
    left, right = np.quantile(upper[:, 0], [0.005, 0.995])
    radius = max(float((right - left) * 0.5), 0.33 * height)
    top = head[head[:, 1] > crown - 0.025 * height]
    centre = np.array([(left + right) * 0.5, crown - radius,
                       float(np.median(top[:, 2]))])
    # Radius comes from skull WIDTH, not its deformed front/back extent.
    # Anchor the sphere under the crown; blend toward the neck and temples.
    up_weight = _smoothstep((p[:, 1] - (centre[1] - 0.95 * radius)) / (0.75 * radius))
    back_weight = _smoothstep((centre[2] - p[:, 2]) / (0.35 * radius))
    skin_weight = _smoothstep((weights - 0.5) / 0.4)
    blend = up_weight * back_weight * skin_weight
    delta = p - centre
    distance = np.linalg.norm(delta, axis=1)
    excess = np.maximum(distance - radius, 0.0)
    amount = blend * excess
    rounded = p - delta * (amount / np.maximum(distance, 1e-12))[:, None]
    if not np.isfinite(rounded).all():
        raise ValueError('Non-finite rounded head geometry')
    return rounded, {
        'centre_raw': centre.tolist(),
        'radius_raw': radius,
        'head_vertices': int(selected.sum()),
        'trimmed_vertices': int((amount > 1e-8).sum()),
        'max_trim_raw': float(amount.max()),
        'full_spherical_cap_vertices': int(((blend > 1.0 - 1e-8) & (excess > 0)).sum()),
    }


def round_human_head(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    skeleton = next((UsdSkel.Skeleton(p) for p in prims if p.IsA(UsdSkel.Skeleton)), None)
    if skeleton is None:
        raise ValueError('Cannot locate the human skeleton for head selection')
    for prim in prims:
        if not prim.IsA(UsdGeom.Mesh) or 'Collision' in prim.GetName():
            continue
        marker = prim.GetAttribute('phyrc:roundedHeadInfo')
        if marker and marker.HasAuthoredValueOpinion():
            return json.loads(marker.Get())
        indices = prim.GetAttribute('primvars:skel:jointIndices')
        weights = prim.GetAttribute('primvars:skel:jointWeights')
        if not indices or not weights:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        joints = UsdSkel.BindingAPI(prim).GetJointsAttr().Get() or skeleton.GetJointsAttr().Get()
        head_joints = [i for i, joint in enumerate(joints)
                       if str(joint).rsplit('/', 1)[-1] in
                       ('head', 'Head', 'jaw', 'left_eye', 'right_eye')]
        ji = np.asarray(indices.Get()).reshape(len(points), -1)
        jw = np.asarray(weights.Get()).reshape(len(points), -1)
        head_weight = np.where(np.isin(ji, head_joints), jw, 0.0).sum(axis=1)
        rounded, info = round_posterior_head(points, head_weight)
        values = Vt.Vec3fArray([Gf.Vec3f(*map(float, point)) for point in rounded])
        mesh.GetPointsAttr().Set(values)
        mesh.CreateExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(values))
        mesh.GetNormalsAttr().Block()
        prim.CreateAttribute('phyrc:roundedHeadInfo', Sdf.ValueTypeNames.String,
                             custom=True).Set(json.dumps(info))
        print('[Head round] ' + json.dumps(info), flush=True)
        return info
    raise ValueError('Cannot locate the skinned visual mesh for head rounding')
