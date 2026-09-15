"""Isaac Sim 6 surface-deformable implementation of the garment helper.

Isaac Sim 6 removed the particle-cloth schemas and ``SingleClothPrim`` used by
the original project. This module keeps the API used by the PhyRC teleop scene
while backing it with the 6.x surface-deformable tensor API.
"""

import os

import numpy as np
import torch
import warp as wp

import omni.kit.commands
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.api import World
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.experimental.prims import DeformablePrim
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import PhysxSchema, Sdf, UsdGeom, UsdShade


def _torch_from_warp(value):
    return wp.to_torch(value)


def _nodal_input(value):
    """Keep live cloth tensors on the GPU; NumPy remains valid for checkpoints."""
    if isinstance(value, torch.Tensor):
        return wp.from_torch(value.detach().to(dtype=torch.float32).contiguous())
    if isinstance(value, wp.array):
        return value
    return np.asarray(value, dtype=np.float32)


class SurfaceClothPrim:
    """Compatibility view exposing the old ClothPrim calls used by PhyRC."""

    supports_particle_masses = False

    def __init__(self, prim_paths_expr=None, prim_path=None, name=None, **_kwargs):
        del name
        self.prim_path = prim_path or prim_paths_expr
        if not self.prim_path:
            raise ValueError("a garment mesh prim path is required")
        self._view = DeformablePrim(self.prim_path, deformable_type="surface")
        self._cloth_prim_view = self
        self.count = 1
        self._device = SimulationManager.get_physics_sim_device()
        self.prim = prims_utils.get_prim_at_path(self.prim_path)

    def initialize(self, *_args, **_kwargs):
        # The experimental view subscribes to SimulationManager's physics-ready
        # event. If constructed after reset, initialize it immediately as well.
        if not self._view.is_physics_tensor_entity_valid():
            SimulationManager._physics_sim_interface.flush_changes()
            self._view._on_physics_ready(None)
        if not self._view.is_physics_tensor_entity_valid():
            raise RuntimeError(f"surface deformable tensor view is not ready: {self.prim_path}")
        return self

    def get_world_positions(self, clone=True):
        # The convenience getter also fetches collision and rest positions and
        # gathers all three arrays. This view represents exactly one surface;
        # only its simulation positions are needed in the physics callbacks.
        assert self.is_physics_tensor_entity_valid(), "surface tensor view is not ready"
        positions = self._view._physics_deformable_body_view.get_simulation_nodal_positions()
        tensor = _torch_from_warp(positions)
        return tensor.clone() if clone else tensor

    def is_physics_tensor_entity_valid(self):
        return self._view.is_physics_tensor_entity_valid()

    def set_world_positions(self, positions):
        self._view.set_nodal_positions(_nodal_input(positions))

    def get_velocities(self, clone=True):
        assert self.is_physics_tensor_entity_valid(), "surface tensor view is not ready"
        tensor = _torch_from_warp(self._view._physics_deformable_body_view.get_simulation_nodal_velocities())
        return tensor.clone() if clone else tensor

    def set_velocities(self, velocities):
        self._view.set_nodal_velocities(_nodal_input(velocities))

    def get_particle_masses(self, clone=True):
        """Return uniform compatibility masses for read-only legacy callers."""
        count = self._view.num_nodes_per_body[0]
        mass = self.prim.GetAttribute("omniphysics:mass").Get() or 0.0
        result = torch.full(
            (1, count),
            float(mass) / max(count, 1),
            dtype=torch.float32,
            device=self._device,
        )
        return result.clone() if clone else result


class Particle_Garment:
    """Build a PhyRC garment with Isaac Sim 6 surface-deformable physics."""

    def __init__(
        self,
        world: World,
        usd_path: str = "Assets/Garment/Dress/Long_LongSleeve/DLLS_Dress008_0/DLLS_Dress008_0_obj.usd",
        pos: np.ndarray = np.array([0.0, 0.0, 0.5]),
        ori: np.ndarray = np.array([0.0, 0.0, 0.0]),
        scale: np.ndarray = np.array([0.0085, 0.0085, 0.0085]),
        visual_material_usd: str = "Assets/Material/Garment/linen_Pumpkin.usd",
        particle_system_enabled: bool = True,
        enable_ccd: bool = True,
        solver_position_iteration_count: int = 16,
        global_self_collision_enabled: bool = True,
        non_particle_collision_enabled: bool = False,
        contact_offset: float = 0.010,
        rest_offset: float = 0.0075,
        particle_contact_offset: float = 0.010,
        fluid_rest_offset: float = 0.0075,
        solid_rest_offset: float = 0.0075,
        adhesion: float = 0.1,
        adhesion_offset_scale: float = 0.0,
        cohesion: float = 0.0,
        particle_adhesion_scale: float = 0.5,
        particle_friction_scale: float = 0.5,
        drag: float = 0.0,
        lift: float = 0.0,
        friction: float = 25.0,
        damping: float = 0.0,
        gravity_scale: float = 1.0,
        particle_mass: float = 1e-2,
        self_collision: bool = True,
        self_collision_filter: bool = True,
        stretch_stiffness: float = 1e12,
        bend_stiffness: float = 100.0,
        shear_stiffness: float = 100.0,
        spring_damping: float = 10.0,
    ):
        # Kept in the signature for source compatibility. Isaac Sim 6 surface
        # deformables do not expose the old particle adhesion/cohesion,
        # per-body gravity, aerodynamic, or self-collision-filter controls.
        del (
            particle_system_enabled,
            global_self_collision_enabled,
            non_particle_collision_enabled,
            fluid_rest_offset,
            adhesion,
            adhesion_offset_scale,
            cohesion,
            particle_adhesion_scale,
            particle_friction_scale,
            drag,
            lift,
            gravity_scale,
            self_collision_filter,
        )

        self.world = world
        self.usd_path = usd_path
        self.position = pos
        self.orientation = ori
        self.scale = scale
        self.visual_material_usd = visual_material_usd
        self.stage = world.stage
        self.scene = world.get_physics_context()._physics_scene

        UsdGeom.Xform.Define(self.stage, "/World/Garment")
        self.garment_name = find_unique_string_name(
            initial_name="garment", is_unique_fn=lambda value: not world.scene.object_exists(value)
        )
        self.garment_prim_path = find_unique_string_name(
            "/World/Garment/garment", is_unique_fn=lambda value: not is_prim_path_valid(value)
        )
        self.deformable_material_path = find_unique_string_name(
            "/World/Garment/deformableMaterial", is_unique_fn=lambda value: not is_prim_path_valid(value)
        )

        add_reference_to_stage(usd_path=self.usd_path, prim_path=self.garment_prim_path)
        self.garment = SingleXFormPrim(
            prim_path=self.garment_prim_path,
            name=self.garment_name,
            position=self.position,
            orientation=euler_angles_to_quat(self.orientation, degrees=True),
            scale=self.scale,
        )

        self.garment_mesh_prim_path = self.garment_prim_path + "/mesh"
        mesh_prim = self.stage.GetPrimAtPath(self.garment_mesh_prim_path)
        if not mesh_prim.IsValid() or not mesh_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"garment mesh did not compose at {self.garment_mesh_prim_path}")
        mesh = UsdGeom.Mesh(mesh_prim)
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        if not face_counts or any(value != 3 for value in face_counts):
            raise RuntimeError(
                f"Isaac Sim 6 surface deformables require triangles: {self.garment_mesh_prim_path}"
            )
        # [quad mesh: refine before defining FEM rest topology]
        from .RefineSurfaceMesh import refine_surface_mesh
        refine_surface_mesh(mesh, int(os.environ.get("STRETCH4_MESH_REFINEMENT", "1")))
        if not deformableUtils.set_physics_surface_deformable_body(
            self.stage, Sdf.Path(self.garment_mesh_prim_path)
        ):
            raise RuntimeError(f"failed to create surface deformable: {self.garment_mesh_prim_path}")

        native_surface_defaults = os.environ.get("STRETCH4_NATIVE_SURFACE", "0") == "1"
        mesh_prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
        # [surface body: explicit velocity damping]
        # Native PhysX coefficient in 1/seconds; independent of material damping.
        linear_damping = float(os.environ.get("STRETCH4_SURFACE_LINEAR_DAMPING", "2"))
        if not np.isfinite(linear_damping) or linear_damping < 0:
            raise ValueError("surface linear damping must be finite and nonnegative")
        mesh_prim.GetAttribute("physxDeformableBody:linearDamping").Set(linear_damping)
        mesh_prim.GetAttribute("physxDeformableBody:selfCollision").Set(bool(self_collision))
        mesh_prim.GetAttribute("physxDeformableBody:enableSpeculativeCCD").Set(bool(enable_ccd))
        mesh_prim.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(
            int(solver_position_iteration_count)
        )
        depenetration_speed = float(os.environ.get("STRETCH4_MAX_DEPENETRATION_VELOCITY", "3.0"))
        if depenetration_speed <= 0:
            raise ValueError("STRETCH4_MAX_DEPENETRATION_VELOCITY must be positive")
        mesh_prim.GetAttribute("physxDeformableBody:maxDepenetrationVelocity").Set(depenetration_speed)
        # Position iterations alone do not refresh contacts during a solve.
        # Under tension the surface can change substantially within one step.
        # These are the native surface-specific contact controls (default 1).
        for attr, value in (
            ("collisionPairUpdateFrequency", int(os.environ.get("STRETCH4_SURFACE_COLLISION_UPDATES", "4"))),
            ("collisionIterationMultiplier", int(os.environ.get("STRETCH4_SURFACE_COLLISION_ITERATIONS", "4"))),
        ):
            if value < 1:
                raise ValueError(f"{attr} must be at least 1")
            mesh_prim.GetAttribute(f"physxDeformableBody:{attr}").Set(value)
        simulation_owner = mesh_prim.GetRelationship("omniphysics:simulationOwner")
        if simulation_owner:
            simulation_owner.SetTargets([self.scene.GetPath()])

        mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
        collision = PhysxSchema.PhysxCollisionAPI(mesh_prim)
        if not native_surface_defaults:
            # Start contacts early while keeping the resting separation small.
            # Contact/rest distances are summed with the rigid shape's offsets;
            # contactOffset is not the thickness of the visible garment.
            surface_contact_offset = float(
                os.environ.get(
                    "STRETCH4_SURFACE_CONTACT_OFFSET", "0.008"
                )
            )
            surface_rest_offset = float(
                os.environ.get(
                    "STRETCH4_SURFACE_REST_OFFSET", "0.005"
                )
            )
            if not 0.0 <= surface_rest_offset < surface_contact_offset:
                raise ValueError("surface contact offset must exceed a nonnegative rest offset")
            collision.GetContactOffsetAttr().Set(surface_contact_offset)
            collision.GetRestOffsetAttr().Set(surface_rest_offset)

        UsdShade.Material.Define(self.stage, self.deformable_material_path)
        surface_friction = min(max(float(os.environ.get("STRETCH4_SURFACE_FRICTION", str(friction))), 0.0), 1.0)
        if native_surface_defaults:
            # Exercise Isaac Sim 6's surface-deformable material model as
            # shipped: density/Young's modulus/Poisson ratio/friction and the
            # 1 mm shell thickness all come from usdLoad/Materials.cpp via
            # NVIDIA's helper. No legacy PBD constants are translated here.
            material_created = deformableUtils.add_surface_deformable_material(
                self.stage, self.deformable_material_path
            )
            print(
                f"[PhyRC 6] native surface defaults: {self.garment_mesh_prim_path} "
                "(thickness=1mm, Young=5e5, Poisson=0.45, dynamic friction=0.25)"
            )
        else:
            # [surface material: thickness compensation]
            # PhysX 110 multiplies surfaceBendStiffness by thickness cubed.
            thickness = float(os.environ.get("STRETCH4_SURFACE_THICKNESS", "0.01"))
            if not np.isfinite(thickness) or thickness <= 0:
                raise ValueError("surface thickness must be finite and positive")
            reference_ratio = 0.005 / thickness
            # Asset metadata preserves total mass and absolute-displacement
            # membrane/bending response under a uniform rest-mesh resize.
            resize_attr = mesh_prim.GetAttribute("phyrc:resizeScale")
            area_attr = mesh_prim.GetAttribute("phyrc:massAreaRatio")
            resize_scale = float(resize_attr.Get()) if resize_attr and resize_attr.HasAuthoredValueOpinion() else 1.0
            mass_area_ratio = float(area_attr.Get()) if area_attr and area_attr.HasAuthoredValueOpinion() else 1.0
            if not np.isfinite([resize_scale, mass_area_ratio]).all() or min(resize_scale, mass_area_ratio) <= 0:
                raise ValueError("invalid garment resize metadata")
            density = float(os.environ.get("STRETCH4_SURFACE_DENSITY", str(44.0 * reference_ratio * mass_area_ratio)))
            # Softer in-plane response for F4 lifting; mass/bending unchanged.
            youngs_modulus = float(os.environ.get("STRETCH4_SURFACE_YOUNGS", str(50000.0 * reference_ratio)))
            poissons_ratio = float(os.environ.get("STRETCH4_SURFACE_POISSON", "0.30"))
            surface_bend_stiffness = float(
                os.environ.get("STRETCH4_SURFACE_BEND", str(2500.0 * reference_ratio ** 3 * resize_scale ** 2))
            )
            material_created = deformableUtils.add_surface_deformable_material(
                self.stage,
                self.deformable_material_path,
                density=density,
                static_friction=surface_friction,
                dynamic_friction=surface_friction,
                youngs_modulus=youngs_modulus,
                poissons_ratio=poissons_ratio,
                surface_thickness=thickness,
                surface_stretch_stiffness=0.0,
                surface_shear_stiffness=0.0,
                surface_bend_stiffness=surface_bend_stiffness,
            )
        if not material_created:
            raise RuntimeError(f"failed to create deformable material: {self.deformable_material_path}")
        material_prim = self.stage.GetPrimAtPath(self.deformable_material_path)
        if not native_surface_defaults:
            material_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
            # These are native 6.x coefficients.  The old PBD damping numbers
            # have different units and must not be copied or scaled implicitly.
            elasticity_damping = float(
                os.environ.get("STRETCH4_SURFACE_ELASTICITY_DAMPING", "0.5")
            )
            bend_damping = float(
                os.environ.get("STRETCH4_SURFACE_BEND_DAMPING", "0.5")
            )
            material_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(
                elasticity_damping
            )
            material_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(bend_damping)
        physicsUtils.add_physics_material_to_prim(
            self.stage, mesh_prim, self.deformable_material_path
        )

        if not native_surface_defaults:
            # Do not turn the old per-particle value into a total body mass: on
            # this 5134-node shirt that made a roughly 51 kg garment.  The 6.x
            # surface density and physical thickness now determine mass.
            print(
                f"[PhyRC 6] fabric surface: {self.garment_mesh_prim_path} "
                f"(thickness={thickness * 1000:.1f}mm, density={density:g}kg/m^3, "
                f"Young={youngs_modulus:g}, Poisson={poissons_ratio:g}, "
                f"contact/rest={surface_contact_offset:g}/{surface_rest_offset:g}m)"
            )

        self.garment_mesh = SurfaceClothPrim(prim_path=self.garment_mesh_prim_path)
        self.particle_controller = self.garment_mesh
        self.particle_system = None

        if self.visual_material_usd is not None:
            self.apply_visual_material(self.visual_material_usd)

    def set_mass(self, mass=0.02):
        prim = self.stage.GetPrimAtPath(self.garment_mesh_prim_path)
        prim.GetAttribute("omniphysics:mass").Set(float(mass))

    def get_particle_system_id(self):
        return 0

    def apply_visual_material(self, material_path: str):
        self.visual_material_path = find_unique_string_name(
            self.garment_prim_path + "/visual_material",
            is_unique_fn=lambda value: not is_prim_path_valid(value),
        )
        add_reference_to_stage(usd_path=material_path, prim_path=self.visual_material_path)
        self.visual_material_prim = prims_utils.get_prim_at_path(self.visual_material_path)
        children = prims_utils.get_prim_children(self.visual_material_prim)
        if not children:
            raise RuntimeError(f"visual material has no child: {material_path}")
        self.material_prim = children[0]
        self.material_prim_path = self.material_prim.GetPath()
        self.visual_material = PreviewSurface(self.material_prim_path)

        targets = [self.garment_mesh_prim_path]
        targets.extend(
            child.GetPath()
            for child in prims_utils.get_prim_children(
                prims_utils.get_prim_at_path(self.garment_mesh_prim_path)
            )
        )
        for target in targets:
            omni.kit.commands.execute(
                "BindMaterialCommand", prim_path=target, material_path=self.material_prim_path
            )

    def get_vertice_positions(self):
        return self.garment_mesh.get_world_positions()[0].detach().cpu().numpy()

    def set_pose(self, pos, ori):
        quat = None if ori is None else euler_angles_to_quat(ori, degrees=True)
        self.garment.set_world_pose(pos, quat)

    def get_particle_system(self):
        return None

    def get_garment_center_pos(self):
        return self.garment.get_world_pose()[0]
