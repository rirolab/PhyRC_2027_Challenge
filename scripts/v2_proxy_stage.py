"""Isaac Sim Stage viewer for exact V2 v0.2 JSONL garment surface grids.

Research-only visualization. Never used as a submitted policy input.
Run INSIDE the existing Isaac Sim GUI via Window > Script Editor:
    import sys
    sys.path.insert(0, '/scripts')
    import v2_proxy_stage
    v2_proxy_stage.show(record=10, which='slack')

Reads the real recorded V2 `surface_grid_world_m`; does not invent mesh shapes.
No PhysX authoring, reset, or simulation timeline writes.
All authored prims go to the USD SESSION layer, not to the saved root layer.
"""

from __future__ import annotations
import json
import math
import os

DEFAULT_JSONL = '/project/research_output/v2_handoff_v02/hdf5_planning_inputs.jsonl'
DEBUG_ROOT = '/World/V2_GARMENT_PROXY_DEBUG'
NAMES = ('taut', 'slack', 'skewed')
COLORS = {'taut': (0.10, 0.44, 0.92),
          'slack': (1.00, 0.48, 0.10),
          'skewed': (0.17, 0.74, 0.32)}
_LAST = {'path': DEFAULT_JSONL, 'record': 10, 'which': 'slack'}


def _as_xyz(x, name):
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise ValueError('%s must be 3-vector' % name)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in x):
        raise ValueError('%s has non-finite / non-numeric values' % name)
    return tuple(float(v) for v in x)


def parse_record(data, require_usd=True):
    """Pure Python input checker, callable without Isaac Sim for unit tests."""
    if data.get('contract') != 'v2-virtual-handoff-0.2':
        raise ValueError('Unexpected V2 contract')
    if require_usd and data.get('kinematics_source') != 'STRETCH4_USD_FK':
        raise ValueError('Refusing synthetic FK as a recorded Stretch4 geometry source')
    if data.get('grasp', {}).get('attachment_mode') != 'assumed_for_test':
        raise ValueError('Unexpected handoff attachment mode')
    grasps = [_as_xyz(p, 'grasp[%d]' % i)
              for i, p in enumerate(data['grasp']['grasp_points_world_m'])]
    if len(grasps) != 2:
        raise ValueError('Expected exactly 2 grasp points')
    poses = data['robot_state']['base_pose_world']
    if len(poses) != 2:
        raise ValueError('Expected exactly 2 base poses')
    bases = []
    for i, pose in enumerate(poses):
        if len(pose) != 7 or not all(isinstance(v, (int,float)) and math.isfinite(v) for v in pose):
            raise ValueError('Invalid robot base pose')
        bases.append(tuple(float(v) for v in pose[:3]))

    scenarios = data['garment_proxy']['scenarios']
    if set(scenarios) != set(NAMES):
        raise ValueError('Exactly taut/slack/skewed required')
    result = {}
    for name in NAMES:
        entry = scenarios[name]
        if entry.get('pass') is not True:
            raise ValueError("%s failed the proxy's own geometric gate" % name)
        surface = entry['surface_grid_world_m']
        landmarks = entry['landmarks_world_m']
        if not isinstance(surface, list) or len(surface) < 3:
            raise ValueError('%s has too few u rows' % name)
        n_u, n_v = len(surface), len(surface[0])
        if n_v < 2 or n_u != len(landmarks):
            raise ValueError('%s grid shape not consistent with landmarks' % name)
        vertices = []
        for u, row in enumerate(surface):
            if len(row) != n_v:
                raise ValueError('%s grid is ragged' % name)
            for v, pt in enumerate(row):
                vertices.append(_as_xyz(pt, '%s surface %s,%s' % (name, u, v)))
            top = _as_xyz(landmarks[u], '%s landmark' % name)
            if _distance(vertices[u*n_v], top) > 1e-6:
                raise ValueError('%s upper mesh boundary differs from landmarks' % name)
        if _distance(vertices[0], grasps[0]) > 1e-6:
            raise ValueError('%s first grasp detached' % name)
        if _distance(vertices[(n_u-1)*n_v], grasps[1]) > 1e-6:
            raise ValueError('%s second grasp detached' % name)
        counts, indices = triangulate(n_u, n_v)
        result[name] = {'n_u': n_u, 'n_v': n_v, 'points': vertices,
                        'counts': counts, 'indices': indices}
    return {'meshes': result, 'grasp_points': grasps, 'base_points': bases,
            'time_s': float(data['timestamp_s']),
            'source': data['kinematics_source'],
            'data_source': data.get('data_source', 'unknown')}


def _distance(a, b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def triangulate(n_u, n_v):
    """Two CCW triangles per cell of rectangular u-major/v-minor grid."""
    if n_u < 2 or n_v < 2:
        raise ValueError('Mesh needs 2x2 vertices')
    counts, indices = [], []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            a = i*n_v+j
            b = (i+1)*n_v+j
            c = (i+1)*n_v+j+1
            d = i*n_v+j+1
            indices.extend((a,b,c,a,c,d))
            counts.extend((3,3))
    return counts, indices


def read_record(path=DEFAULT_JSONL, record=10, require_usd=True):
    if record < 0:
        raise IndexError('record must be >= 0')
    if not os.path.isfile(path):
        raise FileNotFoundError('JSONL missing INSIDE Isaac Sim container: '+path)
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == record:
                return parse_record(json.loads(line), require_usd=require_usd)
    raise IndexError('record %s out of range for %s' % (record, path))


def _stage():
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError('No Isaac Sim USD Stage open')
    return stage


def _stage_session_edit_target(stage):
    from pxr import Usd
    return Usd.EditTarget(stage.GetSessionLayer())


def _ensure_root_is_ours(stage):
    prim = stage.GetPrimAtPath(DEBUG_ROOT)
    if prim.IsValid() and not prim.GetCustomDataByKey('v2_garment_proxy_debug'):
        raise RuntimeError('Refusing to replace existing user prim at '+DEBUG_ROOT)


def _mesh(stage, path, meshdata, rgb, opacity, units_scale):
    from pxr import Gf, UsdGeom
    geom = UsdGeom.Mesh.Define(stage, path)
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geom.CreatePointsAttr([Gf.Vec3f(*(coord * units_scale for coord in p))
                           for p in meshdata['points']])
    geom.CreateFaceVertexCountsAttr(meshdata['counts'])
    geom.CreateFaceVertexIndicesAttr(meshdata['indices'])
    # Ensure accurate bounding box, otherwise viewport framing can miss meshes.
    p = meshdata['points']
    lower = [min(x[a] for x in p) * units_scale for a in range(3)]
    upper = [max(x[a] for x in p) * units_scale for a in range(3)]
    geom.CreateExtentAttr([Gf.Vec3f(*lower), Gf.Vec3f(*upper)])
    geom.CreateDoubleSidedAttr(True)
    geom.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    geom.CreateDisplayOpacityAttr([float(opacity)])
    return geom


def _point(stage, path, pos, rgb, radius_m, units_scale):
    from pxr import Gf, UsdGeom
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius_m*units_scale)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(
        Gf.Vec3d(*(p*units_scale for p in pos)))


def show(record=10, which='slack', path=None, require_usd=True):
    """Insert a chosen JSONL record into the existing Isaac Sim Stage session layer.

    which='taut' | 'slack' | 'skewed' | 'all'. This is ONLY a visualization.
    """
    from pxr import Usd, UsdGeom
    if which not in NAMES + ('all',):
        raise ValueError('which must be taut|slack|skewed|all')
    path = path or DEFAULT_JSONL
    data = read_record(path, record, require_usd=require_usd)
    stage = _stage()
    axis = UsdGeom.GetStageUpAxis(stage)
    if axis != UsdGeom.Tokens.z:
        raise RuntimeError('V2 JSONL uses Z-up world coordinates; current stage is not Z-up')
    metres_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    if not (math.isfinite(metres_per_unit) and metres_per_unit > 0):
        raise RuntimeError('Stage has invalid metersPerUnit')
    units_scale = 1.0/metres_per_unit
    _ensure_root_is_ours(stage)
    old_target = stage.GetEditTarget()
    try:
        stage.SetEditTarget(_stage_session_edit_target(stage))
        # Only remove our own debug namespace in the SESSION layer (not simulation root).
        if stage.GetPrimAtPath(DEBUG_ROOT).IsValid():
            stage.RemovePrim(DEBUG_ROOT)
        root = UsdGeom.Xform.Define(stage, DEBUG_ROOT).GetPrim()
        root.SetCustomDataByKey('v2_garment_proxy_debug', True)
        root.SetCustomDataByKey('source_kind', 'VISUALIZATION_NOT_PHYSICS')
        root.SetCustomDataByKey('record_index', int(record))
        for name, meshdata in data['meshes'].items():
            if which == 'all' or name == which:
                _mesh(stage, DEBUG_ROOT+'/'+name.upper()+'_SURFACE', meshdata,
                      COLORS[name], 0.38 if which=='all' else 0.80, units_scale)
        for i, p in enumerate(data['grasp_points']):
            _point(stage, DEBUG_ROOT+'/GRASP_'+str(i), p, (0.04,0.25,1.0), 0.021, units_scale)
        for i, p in enumerate(data['base_points']):
            _point(stage, DEBUG_ROOT+'/RECORDED_BASE_'+str(i), p,
                   (0.90, 0.12, 0.65), 0.035, units_scale)
    finally:
        stage.SetEditTarget(old_target)
    _LAST.update({'path':path,'record':record,'which':which})
    print('[V2_STAGE] SHOWN record=%d mode=%s t=%.3fs, source=%s, mesh %dx%d, scale=%g' %
          (record, which, data['time_s'], data['source'],
           data['meshes']['taut']['n_u'], data['meshes']['taut']['n_v'], units_scale))
    print('[V2_STAGE] Root: %s | blue=grasp (FK), pink=RECORDED bases (NOT current sim bases)' % DEBUG_ROOT)
    print('[V2_STAGE] NO physics, NO collision, NO state restoration, NO physical grasp verification.')
    return data


def next_frame():
    return show(record=_LAST['record']+1, which=_LAST['which'], path=_LAST['path'])


def previous_frame():
    return show(record=max(0,_LAST['record']-1), which=_LAST['which'], path=_LAST['path'])


def clear():
    """Remove our session-only debug hierarchy, leaving simulation untouched."""
    stage = _stage()
    _ensure_root_is_ours(stage)
    if not stage.GetPrimAtPath(DEBUG_ROOT).IsValid():
        print('[V2_STAGE] Nothing to clear')
        return
    old_target=stage.GetEditTarget()
    try:
        stage.SetEditTarget(_stage_session_edit_target(stage))
        stage.RemovePrim(DEBUG_ROOT)
    finally:
        stage.SetEditTarget(old_target)
    print('[V2_STAGE] Cleared %s from SESSION layer' % DEBUG_ROOT)



def verify_stage():
    """Audit geometry actually authored on the in-memory Isaac Sim USD Stage."""
    from pxr import UsdGeom
    stage=_stage()
    root=stage.GetPrimAtPath(DEBUG_ROOT)
    if not root.IsValid() or not root.GetCustomDataByKey('v2_garment_proxy_debug'):
        raise RuntimeError('No valid V2 debug prim in current Stage')
    report={}
    for name in NAMES:
        prim=stage.GetPrimAtPath(DEBUG_ROOT+'/'+name.upper()+'_SURFACE')
        if not prim.IsValid():
            continue
        mesh=UsdGeom.Mesh(prim)
        np=len(mesh.GetPointsAttr().Get())
        nf=len(mesh.GetFaceVertexCountsAttr().Get())
        ni=len(mesh.GetFaceVertexIndicesAttr().Get())
        collision_schemas=[s for s in prim.GetAppliedSchemas()
                           if 'CollisionAPI' in s or 'RigidBodyAPI' in s or 'Deformable' in s]
        if np != 119 or nf != 192 or ni != 576 or collision_schemas:
            raise RuntimeError('Unexpected geometry or physics schemas: '+str((np,nf,ni,collision_schemas)))
        report[name]={'vertices':np,'triangles':nf,'indices':ni,'physics_schemas':collision_schemas}
    if not report:
        raise RuntimeError('No mesh generated')
    print('[V2_STAGE] STAGE_GEOMETRY_PASS',report)
    print('[V2_STAGE] This is NOT physics/contact/collision PASS.')
    return report
