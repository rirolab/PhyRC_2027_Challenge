import numpy as np
import omni.kit.commands
import omni.physxdemos as demo
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from isaacsim.core.utils.rotations import euler_angles_to_quat


class Human():
    def __init__(self,path,position=None,orientation=None, scale=np.array([0.7, 0.7, 0.7])):
        self.path=path
        self.prim_path="/World/Human"
        add_reference_to_stage(usd_path=path,prim_path=self.prim_path)

        if position is None:
            position=np.array([0,0,0])
        if orientation is None:
            orientation=euler_angles_to_quat([90.0,0.,0.],degrees=True)
        else:
            orientation=euler_angles_to_quat(orientation,degrees=True)
        self.rigid_form=SingleXFormPrim(
            prim_path=self.prim_path,
            name="human",
            position=position,
            orientation=orientation,
            scale=scale,
        )
        
        self.geom_prim=SingleGeometryPrim(
            prim_path=self.prim_path,
            collision=True
        )
        self.geom_prim.set_collision_approximation("meshSimplification")
        self._strip_units_resolve_ops()

    def _strip_units_resolve_ops(self):
        """Safety net: drop Kit's up-axis correction op if one ever appears.

        # [isaac-5.1.0 compat]

        This asset is Y-up and the stage is Z-up, so Kit wants to author its own
        `xformOp:rotateX:unitsResolve = 90` on this prim -- on top of the
        [90,0,0] orientation this class already applies. Two corrections means
        the body lands 180 degrees over, flat on the floor.

        The real fix is upstream of here: run.sh normalizes the staged assets'
        stage metadata (metersPerUnit 1.0, upAxis Z) before Isaac Sim ever opens
        them, so the resolve has nothing to act on. This hook stays as a cheap
        guard in case an asset slips past that pass -- it does no upfront
        waiting, just removes the op if it turns up.
        """
        import omni.kit.app
        from pxr import UsdGeom

        app = omni.kit.app.get_app()
        xformable = UsdGeom.Xformable(self.rigid_form.prim)

        def resolve_ops():
            return [
                op for op in xformable.GetOrderedXformOps()
                if op.GetOpName().endswith(":unitsResolve")
            ]

        # Kit authors the op a few frames after the reference is added, and
        # re-authors it whenever the reference recomposes, so a one-shot strip
        # would not hold. Watch instead, and retire once it has stayed clean.
        self._resolve_clean_frames = 0

        def _keep_clean(_event):
            back = resolve_ops()
            if back:
                self._resolve_clean_frames = 0
                remaining = [
                    op for op in xformable.GetOrderedXformOps()
                    if not op.GetOpName().endswith(":unitsResolve")
                ]
                xformable.SetXformOpOrder(remaining, resetXformStack=False)
                for op in back:
                    self.rigid_form.prim.RemoveProperty(op.GetOpName())
                return
            self._resolve_clean_frames += 1
            if self._resolve_clean_frames > 240:
                self._resolve_sub = None

        self._resolve_sub = app.get_update_event_stream().create_subscription_to_pop(
            _keep_clean, name="strip_units_resolve"
        )
