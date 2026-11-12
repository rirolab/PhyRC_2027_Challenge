"""V2 PhysX/Warp garment overlay for Isaac Sim 6.0.1 (GPU physics).

Does not control robots, modify original USD assets, or simulate cloth.
The simulation view is created with the ACTIVE stage_id; this is essential
when the application has an attached Stage and a GPU physics pipeline.

In Isaac Sim Script Editor (after Play):
    import sys; sys.path.insert(0, '/scripts')
    import importlib, v2_physx_tracking as phys
    importlib.reload(phys)
    phys.inspect()
    phys.start(which='slack', every=3)
    phys.status()
    phys.stop(clear_overlay=True)
"""
import math
import traceback

PATHS = ('/World/Stretch4/grasp_center_link',
         '/World/Stretch4_2/grasp_center_link')
_LIVE = None


def _vec3(x):
    values = tuple(float(v) for v in x)
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise RuntimeError('PHYSICS_SAMPLE_INVALID: xyz')
    return values


def _to_numpy(value):
    """Convert the OUTPUT of a Warp GPU tensor to CPU for small-scale diagnosis.

    This does NOT select the incompatible NumPy *tensor frontend*.
    """
    if hasattr(value, 'numpy'):
        return value.numpy()
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    raise RuntimeError('UNSUPPORTED_TENSOR_RESULT: expected Warp or Torch tensor')


def _physics_stage():
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, UsdGeom
    if not omni.timeline.get_timeline_interface().is_playing():
        raise RuntimeError('PHYSICS_NOT_PLAYING: press Play, then rerun')
    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError('NO_USD_STAGE')
    stage_id = context.get_stage_id()
    if stage_id is None:
        raise RuntimeError('NO_ACTIVE_STAGE_ID')
    for path in PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError('MISSING_RIGID_LINK: ' + path)
    units = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(units) or units <= 0:
        raise RuntimeError('BAD_STAGE_UNITS')
    return context, stage, stage_id, units


class PhysxTensorSource:
    """Read live world poses of exactly two physics rigid bodies.

    Do not fall back to USD transforms. World positions are in SI meters.
    """
    def __init__(self):
        context, stage, stage_id, self.meters_per_unit = _physics_stage()
        import omni.physics.tensors as tensors
        self.stage_id = stage_id
        self.stage_identifier = stage.GetRootLayer().identifier
        self.paths = list(PATHS)
        # FIX: the previous bridge omitted stage_id. Its autonomous test PASSed
        # with the active stage_id on this user's GPU physics configuration.
        self.sim = tensors.create_simulation_view('warp', stage_id=stage_id)
        self.sim.set_subspace_roots('/')
        self.views = []
        for path in self.paths:
            body = self.sim.create_rigid_body_view(path)
            if int(body.count) != 1:
                raise RuntimeError('TENSOR_RIGID_BODY_COUNT: %s count=%s' % (path, body.count))
            self.views.append(body)
        self.read_poses()  # fail before modifying Stage

    def read_poses(self):
        _physics_stage()  # fail closed if simulation stops
        records = []
        for path, body in zip(self.paths, self.views):
            values = _to_numpy(body.get_transforms()).reshape(-1, 7)
            if values.shape != (1, 7):
                raise RuntimeError('TENSOR_POSE_SHAPE: %s %s' % (path, values.shape))
            xyz = _vec3(values[0, :3])
            quat = tuple(float(v) for v in values[0, 3:7])
            if not all(math.isfinite(v) for v in quat) or abs(sum(v*v for v in quat)-1) > .02:
                raise RuntimeError('TENSOR_QUATERNION_INVALID: ' + path)
            records.append({'path': path, 'xyz_m': xyz, 'quaternion_xyzw': quat})
        return records

    def read(self):
        return [record['xyz_m'] for record in self.read_poses()]


def _anchor_error(points, scenario, meters_per_unit):
    from pxr import UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    path = '/World/V2_GARMENT_PROXY_DEBUG/' + scenario.upper() + '_SURFACE'
    mesh = UsdGeom.Mesh.Get(stage, path)
    if not mesh or not mesh.GetPrim().IsValid():
        raise RuntimeError('MESH_MISSING: ' + path)
    vertices = mesh.GetPointsAttr().Get()
    if vertices is None or len(vertices) != 119:
        raise RuntimeError('INVALID_MESH_POINTS: expected 119')
    a = _vec3([v * meters_per_unit for v in vertices[0]])
    b = _vec3([v * meters_per_unit for v in vertices[16*7]])
    return max(math.dist(a, points[0]), math.dist(b, points[1]))


class Tracking:
    def __init__(self, source, which, every, movement_threshold, anchor_tolerance):
        self.source = source
        self.which = which
        self.every = every
        self.move_threshold_m = float(movement_threshold)
        self.anchor_tolerance_m = float(anchor_tolerance)
        self.first = source.read()
        self.last = self.first
        self.max_move = [0.0, 0.0]
        self.max_anchor_error = 0.0
        self.physics_samples = 0
        self.overlay_updates = 0
        self.ticks = 0
        self.error = None
        self.sub = None
        self.scale = None

    def start(self):
        import v2_live_proxy as overlay
        # Make sure the old USD-preview callback cannot race with the PhysX one.
        overlay.stop(clear_overlay=False)
        self.scale = overlay._safe_create('physics', self.which)
        self._sample()  # initialization must work before subscribing
        import omni.kit.app
        self.sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._tick, name='V2 PhysX-only proxy tracking')
        print('[V2_PHYSX] START source=PHYSICS_TENSORS_WARP stage_id=%s' % self.source.stage_id)

    def _sample(self):
        import v2_live_proxy as overlay
        pts = self.source.read()  # real physics poses, one read per update
        overlay._write_meshes(pts, self.which, self.scale)
        for i in range(2):
            self.max_move[i] = max(self.max_move[i], math.dist(self.first[i], pts[i]))
        names = ('taut', 'slack', 'skewed') if self.which == 'all' else (self.which,)
        error = max(_anchor_error(pts, name, self.source.meters_per_unit) for name in names)
        self.max_anchor_error = max(self.max_anchor_error, error)
        if error > self.anchor_tolerance_m:
            raise RuntimeError('ANCHOR_MISMATCH: %.6fm > %.6fm' % (error, self.anchor_tolerance_m))
        self.last = pts
        self.physics_samples += 1
        self.overlay_updates += 1

    def _tick(self, _event):
        self.ticks += 1
        if self.ticks % self.every:
            return
        try:
            import omni.usd
            if omni.usd.get_context().get_stage_id() != self.source.stage_id:
                raise RuntimeError('ACTIVE_STAGE_CHANGED')
            self._sample()
            if self.physics_samples % 90 == 0:
                print('[V2_PHYSX] samples=%d movement=%s anchor_error=%.6f' % (
                    self.physics_samples, self.max_move, self.max_anchor_error))
        except Exception as exc:
            self.error = type(exc).__name__ + ': ' + str(exc)
            self.sub = None
            print('[V2_PHYSX] STOP_ON_ERROR', self.error)
            traceback.print_exc()

    def report(self):
        moved = all(x >= self.move_threshold_m for x in self.max_move)
        running = self.sub is not None
        passed = running and self.error is None and self.physics_samples >= 5 and moved \
            and self.max_anchor_error <= self.anchor_tolerance_m
        return {
            'source': 'PHYSICS_TENSORS_WARP',
            'stage_id': self.source.stage_id,
            'running': running,
            'physics_samples_checked': self.physics_samples,
            'overlay_updates': self.overlay_updates,
            'movement_m_per_robot': list(self.max_move),
            'movement_threshold_m': self.move_threshold_m,
            'max_anchor_error_m': self.max_anchor_error,
            'anchor_tolerance_m': self.anchor_tolerance_m,
            'last_points_world_m': self.last,
            'error': self.error,
            'result': 'LIVE_PHYSICS_TRACKING_PASS' if passed else ('FAIL' if self.error else 'NOT_YET_PROVEN'),
            'cloth_physics_verified': False,
            'physical_attachment_verified': False,
        }


def inspect():
    try:
        src = PhysxTensorSource()
        poses = src.read_poses()
        print('[V2_PHYSX] TENSOR_HANDLES_PASS stage_id=%s poses=%s' % (src.stage_id, poses))
        return True
    except Exception as exc:
        print('[V2_PHYSX] TENSOR_HANDLES_NOT_READY:', type(exc).__name__, str(exc))
        traceback.print_exc()
        return False


def start(which='slack', every=3, movement_threshold_m=.005, anchor_tolerance_m=.001):
    global _LIVE
    if which not in ('taut', 'slack', 'skewed', 'all'):
        raise ValueError('Unknown scenario')
    if not isinstance(every, int) or every < 1:
        raise ValueError('every must be integer >=1')
    if _LIVE is not None:
        stop(clear_overlay=False)
    source = PhysxTensorSource()
    tracking = Tracking(source, which, every, movement_threshold_m, anchor_tolerance_m)
    tracking.start()
    _LIVE = tracking
    return status()


def status():
    result = {'running': False, 'result': 'NOT_STARTED'} if _LIVE is None else _LIVE.report()
    print('[V2_PHYSX] STATUS', result)
    return result


def stop(clear_overlay=False):
    global _LIVE
    if _LIVE is not None:
        _LIVE.sub = None
        _LIVE = None
    if clear_overlay:
        import v2_live_proxy as overlay
        overlay.clear()
    print('[V2_PHYSX] STOPPED')
