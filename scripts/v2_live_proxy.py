"""Live world-frame garment proxy overlay for an existing Isaac Sim GUI session.

STRICTLY VISUAL: no robot commands, PhysX schema, timeline controls, or USD root writes.
Runtime mode reads dynamic-control physics rigid-body poses for grasp_center_link.
A separately named 'usd_preview' mode can read stage XformCache; it is NOT
proof of live PhysX state (Fabric may make USD transforms stale).

Script Editor:
    import sys
    sys.path.insert(0, '/scripts')
    import v2_live_proxy as live
    live.inspect()
    live.start(mode='physics', which='slack')
    live.status()
    live.stop()
"""
from __future__ import annotations

import math
import time
import traceback

ROOT = '/World/V2_GARMENT_PROXY_DEBUG'
ROBOTS = ('/World/Stretch4', '/World/Stretch4_2')
LINK = 'grasp_center_link'
SCENARIOS = ('taut', 'slack', 'skewed')
COLORS = {'taut': (0.10, 0.44, 0.92), 'slack': (1.00, 0.48, 0.10), 'skewed': (0.15, 0.74, 0.32)}
SAG = {'taut': 0.015, 'slack': 0.14, 'skewed': 0.06}
SIDE = {'taut': 0., 'slack': 0.02, 'skewed': 0.08}
_RUNTIME = None


def _check_xyz(p):
    if len(p) != 3:
        raise ValueError('Expected xyz 3-vector')
    out = tuple(float(a) for a in p)
    if not all(math.isfinite(a) for a in out):
        raise ValueError('Nonfinite world coordinates')
    return out


def _norm(p):
    return math.sqrt(sum(v*v for v in p))


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def triangulate(n_u=17, n_v=7):
    if n_u < 2 or n_v < 2:
        raise ValueError('Invalid grid')
    counts, indices = [], []
    for i in range(n_u-1):
        for j in range(n_v-1):
            a=i*n_v+j; b=(i+1)*n_v+j; c=b+1; d=a+1
            indices.extend((a,b,c,a,c,d))
            counts.extend((3,3))
    return counts, indices


def build_proxy(grasp_points, which='slack', n_u=17, n_v=7, depth_m=.28):
    """Same kinematic 17x7 surface formula as V2 v0.2 garment.py.

    Not a shirt mesh, no elastic forces or attachment verification.
    Returns *meters* in the SAME stage world frame as supplied points.
    """
    if which not in SCENARIOS:
        raise ValueError('Unknown scenario')
    a,b=tuple(_check_xyz(x) for x in grasp_points)
    delta=_sub(b,a)
    span=_norm(delta)
    if span <= 1e-7:
        raise ValueError('Grasp points coincident')
    tangent=tuple(x/span for x in delta)
    side=_cross(tangent,(0.,0.,1.))
    if _norm(side)<1e-8:
        side=_cross(tangent,(0.,1.,0.))
    m=_norm(side)
    side=tuple(x/m for x in side)
    vertices=[]
    top=[]
    for i in range(n_u):
        u=i/(n_u-1)
        bell=math.sin(math.pi*u)
        pt=tuple(a[k] + u*delta[k] + bell*(SIDE[which]*side[k] - (SAG[which] if k==2 else 0.)) for k in range(3))
        if i==0: pt=a
        if i==n_u-1: pt=b
        top.append(pt)
        for j in range(n_v):
            v=j/(n_v-1)
            extra=.3*SIDE[which]*math.sin(math.pi*u)*math.sin(math.pi*v)
            vertices.append(tuple(pt[k] - (v*depth_m if k==2 else 0.) + extra*side[k] for k in range(3)))
    assert vertices[0] == a and vertices[(n_u-1)*n_v] == b
    return {'points': vertices, 'landmarks': top, 'span_m': span,
            'grid': (n_u,n_v), 'counts': triangulate(n_u,n_v)[0],
            'indices': triangulate(n_u,n_v)[1]}


def _stage():
    import omni.usd
    s=omni.usd.get_context().get_stage()
    if s is None:
        raise RuntimeError('No stage')
    return s


def _paths():
    """Return actual stage paths, never assume missing descendants exist."""
    stage=_stage()
    paths=[]
    for root in ROBOTS:
        r=stage.GetPrimAtPath(root)
        if not r.IsValid():
            raise RuntimeError('Missing robot prim '+root)
        options=[str(p.GetPath()) for p in r.GetAllChildren() if p.GetName()==LINK]
        # Recursive traversal; only within robot subtree.
        stack=[r]
        while stack:
            prim=stack.pop()
            if prim.GetName()==LINK:
                options.append(str(prim.GetPath()))
            stack.extend(prim.GetChildren())
        options=list(dict.fromkeys(options))
        if len(options)!=1:
            raise RuntimeError('%s expected ONE %s link, found %s' % (root,LINK,options))
        paths.append(options[0])
    return paths


class _Source:
    def __init__(self, mode):
        if mode not in ('physics','usd_preview'):
            raise ValueError("mode must be 'physics' or 'usd_preview'")
        self.mode=mode
        self.paths=_paths()
        self.dc=None
        self.rb=None
        self.art=[]
        if mode=='physics':
            from omni.isaac.dynamic_control import _dynamic_control
            self.dc=_dynamic_control.acquire_dynamic_control_interface()
            self.rb=[]
            for path in self.paths:
                h=self.dc.get_rigid_body(path)
                if not h:
                    raise RuntimeError('Physics rigid body not available at '+path+'; '
                                       'cannot claim PHYSICS_LIVE. Try live.inspect(), '
                                       'or usd_preview (unverified) if appropriate.')
                self.rb.append(h)
            # Articulations may live at a nested ArticulationRootAPI prim.
            # Failure to locate joint metadata does not change measured rigid-body poses.
            try:
                from pxr import UsdPhysics
                stage=_stage()
                for root_path in ROBOTS:
                    candidates=[root_path]
                    stack=[stage.GetPrimAtPath(root_path)]
                    while stack:
                        prim=stack.pop()
                        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                            candidates.append(str(prim.GetPath()))
                        stack.extend(prim.GetChildren())
                    art_handle=None
                    for candidate in dict.fromkeys(candidates):
                        a=self.dc.get_articulation(candidate)
                        if a:
                            art_handle=a
                            break
                    self.art.append(art_handle)
            except Exception as e:
                self.art=[None,None]
                print('[V2_LIVE] Joint inspection unavailable:',type(e).__name__,str(e))

    def read_joint_state(self):
        if self.mode!='physics':
            return {'available':False,'reason':'USD preview is not verified live joint state'}
        robots=[]
        for index,art in enumerate(self.art):
            if not art:
                robots.append({'robot':index,'available':False,'reason':'No articulation handle'})
                continue
            count=self.dc.get_articulation_dof_count(art)
            joints=[]
            for i in range(count):
                h=self.dc.get_articulation_dof(art,i)
                joints.append({'name':self.dc.get_dof_name(h),
                               'position':float(self.dc.get_dof_position(h))})
            robots.append({'robot':index,'available':True,'joints':joints})
        return {'available':all(r['available'] for r in robots),'robots':robots}

    def read(self):
        if self.mode=='physics':
            out=[]
            for h in self.rb:
                p=self.dc.get_rigid_body_pose(h).p
                out.append(_check_xyz((p.x,p.y,p.z)))
            return out
        from pxr import UsdGeom
        stage=_stage()
        cache=UsdGeom.XformCache()
        scale=UsdGeom.GetStageMetersPerUnit(stage)
        out=[]
        for path in self.paths:
            mat=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            p=mat.ExtractTranslation()
            out.append(_check_xyz((p[0]*scale,p[1]*scale,p[2]*scale)))
        return out


def _session():
    from pxr import Usd
    return Usd.EditTarget(_stage().GetSessionLayer())


def _assert_debug_namespace(stage):
    prim=stage.GetPrimAtPath(ROOT)
    if prim.IsValid() and prim.GetCustomDataByKey('v2_garment_proxy_debug') is not True:
        raise RuntimeError('Refusing to touch non-V2 namespace '+ROOT)


def _safe_create(mode, which):
    from pxr import Gf, UsdGeom
    stage=_stage()
    _assert_debug_namespace(stage)
    if UsdGeom.GetStageUpAxis(stage)!=UsdGeom.Tokens.z:
        raise RuntimeError('Stage not Z-up')
    metres_per_unit=UsdGeom.GetStageMetersPerUnit(stage)
    if metres_per_unit <= 0:
        raise RuntimeError('Invalid stage units')
    scale=1/metres_per_unit
    target=stage.GetEditTarget()
    try:
        stage.SetEditTarget(_session())
        # Replace only own namespace; old HDF5 preview overlay gets replaced.
        if stage.GetPrimAtPath(ROOT).IsValid():
            stage.RemovePrim(ROOT)
        root=UsdGeom.Xform.Define(stage,ROOT).GetPrim()
        root.SetCustomDataByKey('v2_garment_proxy_debug',True)
        root.SetCustomDataByKey('source_kind', 'LIVE_PHYSICS_READ_ONLY' if mode=='physics' else 'USD_PREVIEW_UNVERIFIED')
        root.SetCustomDataByKey('physical_attachment_verified',False)
        names=SCENARIOS if which=='all' else (which,)
        for name in names:
            mesh=UsdGeom.Mesh.Define(stage, ROOT+'/'+name.upper()+'_SURFACE')
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateFaceVertexCountsAttr(triangulate()[0])
            mesh.CreateFaceVertexIndicesAttr(triangulate()[1])
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*COLORS[name])])
            mesh.CreateDisplayOpacityAttr([.40 if which=='all' else .82])
        for i in range(2):
            sphere=UsdGeom.Sphere.Define(stage,ROOT+'/GRASP_'+str(i))
            sphere.CreateRadiusAttr(.025*scale)
            sphere.CreateDisplayColorAttr([Gf.Vec3f(.03,.35,.99)])
    finally:
        stage.SetEditTarget(target)
    return scale


def _write_meshes(points, which, scale):
    from pxr import Gf, UsdGeom
    stage=_stage()
    _assert_debug_namespace(stage)
    target=stage.GetEditTarget()
    out={}
    try:
        stage.SetEditTarget(_session())
        for name in SCENARIOS if which=='all' else (which,):
            data=build_proxy(points,name)
            mesh=UsdGeom.Mesh(stage.GetPrimAtPath(ROOT+'/'+name.upper()+'_SURFACE'))
            pts=[Gf.Vec3f(*(v*scale for v in p)) for p in data['points']]
            mesh.GetPointsAttr().Set(pts)
            lower=[min(p[k] for p in data['points'])*scale for k in range(3)]
            upper=[max(p[k] for p in data['points'])*scale for k in range(3)]
            mesh.GetExtentAttr().Set([Gf.Vec3f(*lower),Gf.Vec3f(*upper)])
            out[name]={'vertices':len(pts),'triangles':len(data['counts']), 'span_m':data['span_m']}
        for i,p in enumerate(points):
            sphere=UsdGeom.Sphere(stage.GetPrimAtPath(ROOT+'/GRASP_'+str(i)))
            UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(Gf.Vec3d(*(v*scale for v in p)))
    finally:
        stage.SetEditTarget(target)
    return out


class _Tracker:
    def __init__(self, mode, which, every):
        if which not in SCENARIOS+('all',): raise ValueError('Invalid scenario')
        if not isinstance(every,int) or every<1: raise ValueError('every>=1 expected')
        self.source=_Source(mode)
        self.mode=mode; self.which=which; self.every=every
        self.scale=None; self.sub=None
        self.ticks=0; self.updates=0; self.last=None; self.failed=None
        self.first=None; self.max_drift=0.

    def begin(self):
        points=self.source.read() # refuse to mutate stage if no valid sensor pose
        self.scale=_safe_create(self.mode,self.which)
        _write_meshes(points,self.which,self.scale)
        self.first=points;self.last=points;self.updates=1
        import omni.kit.app
        stream=omni.kit.app.get_app().get_update_event_stream()
        self.sub=stream.create_subscription_to_pop(self._tick,name='V2 Live Garment Proxy (read-only)')
        print('[V2_LIVE] START mode=%s which=%s grasp_links=%s' %(self.mode,self.which,self.source.paths))
        print('[V2_LIVE] first_points_world_m=%s' %points)
        print('[V2_LIVE] THIS IS VISUALIZATION ONLY: grasp attachment NOT validated.')

    def _tick(self,event):
        self.ticks+=1
        if self.ticks%self.every: return
        try:
            points=self.source.read()
            _write_meshes(points,self.which,self.scale)
            self.last=points; self.updates+=1
            drift=max(_norm(_sub(a,b)) for a,b in zip(self.first,points))
            self.max_drift=max(self.max_drift,drift)
            if self.updates%90==0:
                print('[V2_LIVE] updates=%d drift_max=%.4fm mode=%s' %(self.updates,self.max_drift,self.mode))
        except Exception as e:
            self.failed='%s: %s'%(type(e).__name__,e)
            self.sub=None  # callback subscription object freed => stop; avoid repeated exceptions
            print('[V2_LIVE] STOP ON ERROR: '+self.failed)
            traceback.print_exc()

    def stop(self):
        self.sub=None
        print('[V2_LIVE] STOP updates=%d max_grasp_displacement_m=%.5f' %(self.updates,self.max_drift))


def inspect():
    """Discover grasp-center prim paths and runtime physics handle availability."""
    stage=_stage()
    print('[V2_LIVE] Stage:',stage.GetRootLayer().identifier)
    from pxr import UsdGeom
    print('[V2_LIVE] Stage units metersPerUnit:',UsdGeom.GetStageMetersPerUnit(stage))
    paths=_paths()
    print('[V2_LIVE] Grasp link prims:',paths)
    try:
        src=_Source('physics')
        print('[V2_LIVE] PHYSICS rigid-body handles AVAILABLE; points:',src.read())
    except Exception as e:
        print('[V2_LIVE] PHYSICS unavailable:',type(e).__name__,str(e))
    try:
        src=_Source('usd_preview')
        print('[V2_LIVE] USD transform preview (NOT physical source):',src.read())
    except Exception as e:
        print('[V2_LIVE] USD preview unavailable:',type(e).__name__,str(e))
    return paths


def robot_state():
    """Read current grasp poses and articulation joint positions without any commands."""
    t=_RUNTIME
    if t is None:
        raise RuntimeError('Call live.start() first')
    state={'grasp_points_world_m':t.source.read(),
           'source': 'PHYSICS_RIGID_BODY_POSE' if t.mode=='physics' else 'USD_STAGE_PREVIEW_UNVERIFIED',
           'joint_state': t.source.read_joint_state(),
           'physical_attachment_verified':False}
    print('[V2_LIVE] ROBOT_STATE',state)
    return state


def start(mode='physics', which='slack', every=3):
    """Start read-only live tracking. No robot motion, no timeline manipulation."""
    global _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME.stop();_RUNTIME=None
    t=_Tracker(mode,which,every)
    t.begin()
    _RUNTIME=t
    return status()


def status():
    t=_RUNTIME
    if t is None:
        d={'running':False}
    else:
        d={'running': t.sub is not None, 'mode':t.mode, 'which':t.which,
           'source':'PHYSICS_RIGID_BODY_POSE' if t.mode=='physics' else 'USD_STAGE_PREVIEW_UNVERIFIED',
           'ticks':t.ticks,'updates':t.updates,'last_points_world_m':t.last,
           'max_grasp_displacement_m':t.max_drift,'error':t.failed,
           'physical_attachment_verified':False, 'physics_garment_simulated':False}
    print('[V2_LIVE] STATUS',d)
    return d


def stop(clear_overlay=False):
    global _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME.stop();_RUNTIME=None
    if clear_overlay:
        clear()


def clear():
    """Clear our debug namespace from session layer only."""
    stage=_stage()
    _assert_debug_namespace(stage)
    if not stage.GetPrimAtPath(ROOT).IsValid():return
    target=stage.GetEditTarget()
    try:
        stage.SetEditTarget(_session())
        stage.RemovePrim(ROOT)
    finally:
        stage.SetEditTarget(target)
    print('[V2_LIVE] Debug overlay cleared')
