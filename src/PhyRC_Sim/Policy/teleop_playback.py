"""USD state playback. Physics schemas are disabled in the session layer."""
import numpy as np


def disable_physics(stage):
    """Remove physics from an in-memory copy before Kit can start cooking it."""
    from pxr import Sdf
    stage.SetEditTarget(stage.GetSessionLayer())
    while True:
        instances = [p for p in stage.Traverse() if p.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    for prim in list(stage.Traverse()):
        if prim.GetTypeName().startswith(('Physics', 'Physx', 'OmniPhysics')):
            prim.SetActive(False)
        else:
            authored = prim.GetMetadata('apiSchemas')
            apis = authored.GetAppliedItems() if authored else []
            apis = [a for a in apis if not a.startswith(('Physics', 'Physx', 'OmniPhysics'))]
            prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(apis))


class ReplayScene:
    def __init__(self, stage, metadata, follow_camera=True):
        from pxr import Gf, UsdGeom
        self.stage, self.metadata = stage, metadata
        self.follow_camera = follow_camera
        disable_physics(stage)
        self.link_ops = []
        for paths in metadata['robot_link_paths']:
            operations = []
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    raise ValueError(f'Recorded link missing from scene: {path}')
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                position = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, 'fullReplay')
                orientation = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, 'fullReplay')
                scale = xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, 'fullReplay')
                xf.SetResetXformStack(True)
                operations.append((position, orientation, scale))
            self.link_ops.append(operations)
        self.meshes = []
        for path in metadata['garment_paths']:
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
            if not mesh:
                raise ValueError(f'Recorded garment missing from scene: {path}')
            xf = UsdGeom.Xformable(mesh)
            xf.MakeMatrixXform().Set(Gf.Matrix4d(1))
            xf.SetResetXformStack(True)
            # Rendering should recompute normals for each recorded deformation.
            mesh.GetNormalsAttr().Block()
            self.meshes.append(mesh)
        self.static = {}
        for name in ('Human', 'Chair'):
            xf = UsdGeom.Xformable(stage.GetPrimAtPath('/World/' + name))
            self.static[name] = xf.MakeMatrixXform()
            xf.SetResetXformStack(True)
        self.camera = UsdGeom.Camera.Define(stage, '/World/FullReplayCamera')
        self.camera_op = UsdGeom.Xformable(self.camera).MakeMatrixXform()
        UsdGeom.Xformable(self.camera).SetResetXformStack(True)

    def apply(self, values):
        from pxr import Gf, Vt
        for i, operations in enumerate(self.link_ops):
            poses = values[f'r{i}_links']  # Physics convention: xyz, xyzw.
            if len(poses) != len(operations):
                raise ValueError('Recorded link layout differs from scene')
            for j, (position, orientation, scale) in enumerate(operations):
                raw = poses[j]
                position.Set(Gf.Vec3d(*map(float, raw[:3])))
                orientation.Set(Gf.Quatd(float(raw[6]), Gf.Vec3d(*map(float, raw[3:6]))))
                scale.Set(Gf.Vec3d(*self.metadata['robot_link_scales'][i][j]))
        for i, mesh in enumerate(self.meshes):
            points = np.ascontiguousarray(values[f'g{i}_positions'], dtype=np.float32)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
            lo, hi = points.min(0), points.max(0)
            mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([lo, hi])))
        for name, op in self.static.items():
            op.Set(Gf.Matrix4d(values[f'static_{name}'].tolist()))
        if self.follow_camera and 'camera_world' in values:
            self.camera_op.Set(Gf.Matrix4d(values['camera_world'].tolist()))
            focal, horizontal, vertical, near, far = values['camera_lens']
            self.camera.GetFocalLengthAttr().Set(float(focal))
            self.camera.GetHorizontalApertureAttr().Set(float(horizontal))
            self.camera.GetVerticalApertureAttr().Set(float(vertical))
            self.camera.GetClippingRangeAttr().Set(Gf.Vec2f(float(near), float(far)))

    def verify(self, values):
        """Exact stored data readback, plus world-space transform validation."""
        from pxr import Usd, UsdGeom
        for i, mesh in enumerate(self.meshes):
            if not np.array_equal(np.asarray(mesh.GetPointsAttr().Get()), values[f'g{i}_positions']):
                raise AssertionError('Cloth replay differs from recorded vertices')
            world = np.asarray(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            if not np.array_equal(world, np.eye(4)):
                raise AssertionError('Cloth replay vertices are not in world coordinates')
        for i, operations in enumerate(self.link_ops):
            for j, (position, orientation, _) in enumerate(operations):
                raw = values[f'r{i}_links'][j]
                q = orientation.Get()
                if not np.array_equal(np.array(position.Get()), raw[:3]):
                    raise AssertionError('Robot replay translation differs from recording')
                if not np.array_equal(np.r_[q.GetImaginary(), q.GetReal()], raw[3:]):
                    raise AssertionError('Robot replay orientation differs from recording')
                prim = self.stage.GetPrimAtPath(self.metadata['robot_link_paths'][i][j])
                matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                if not np.array_equal(matrix[3, :3], raw[:3]):
                    raise AssertionError('Replay hierarchy changed a world-space robot pose')
        for name, op in self.static.items():
            if not np.array_equal(np.asarray(op.Get()), values[f'static_{name}']):
                raise AssertionError('Static placement differs from recording')
