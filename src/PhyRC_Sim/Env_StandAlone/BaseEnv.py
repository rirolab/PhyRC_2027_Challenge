from isaacsim import SimulationApp
# simulation_app = SimulationApp({"headless": False})

import os
import sys
import numpy as np
import pickle
from termcolor import cprint

import omni.kit.commands
import omni.replicator.core as rep
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.api import World
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path, get_prim_children
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, update_stage
from isaacsim.core.prims import XFormPrim, RigidPrim, GeometryPrim
from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade


sys.path.append(os.getcwd())
from Env_Config.Utils_Project.Code_Tools import get_unique_filename


class BaseEnv:
    def __init__(self) -> None:
        # set world
        self.world = World()
        # [isaac-5.1.0 compat: physics rate]
        import os as _os
        # 360 Hz -- a compromise. 480 was measurably calmer but the slow
        # motion was too much to work in; 240 was where the tunnelling
        # collision checks, which is the direct lever against a thin feature
        # slipping through the cloth -- the fingertip measures 0.0047m against
        # a 0.0031m gap between particle spheres, so there is not much margin
        # to work with and the step size has to make up the difference. The
        # cost is that everything runs at half speed; that trade was accepted
        # deliberately ("stability matters more, even if it runs in slow motion").
        _hz = float(_os.environ.get('STRETCH4_PHYSICS_HZ', '240'))
        if _hz != 60.0:
            self.world.set_simulation_dt(physics_dt=1.0 / _hz,
                                         rendering_dt=1.0 / 60.0)
        # set scene
        self.scene = self.world.scene
        # set stage
        self.stage = self.world.scene.stage
        # set simulation context
        self.context = SimulationContext()
        # set physics context
        self.physics = self.world.get_physics_context()
        # set physics scene
        self.physics.enable_ccd(True)
        # [isaac-5.1.0 compat: enhanced determinism]
        if _os.environ.get('STRETCH4_DETERMINISM', '1') == '1':
            try:
                from pxr import PhysxSchema as _PxS
                _sc = _PxS.PhysxSceneAPI.Apply(
                    self.stage.GetPrimAtPath('/physicsScene'))
                _sc.CreateEnableEnhancedDeterminismAttr().Set(True)
                print('[BaseEnv] enhanced determinism requested', flush=True)
            except Exception as _e:
                print(f'[BaseEnv] enhanced determinism unavailable: {_e}',
                      flush=True)
        self.physics.enable_gpu_dynamics(True)
        # Isaac Sim 6 validates the token case (GPU/MBP/SAP).
        self.physics.set_broadphase_type("GPU")
        self.physics.enable_stablization(True)
        self.physics.set_solver_type("TGS")
        self.physics.set_gpu_max_rigid_contact_count(10240000)
        self.physics.set_gpu_max_rigid_patch_count(10240000)
        # Surface deformables have their own GPU contact pool in PhysX 5.6.
        # Four dense shirts can exceed the small default during their initial
        # table contact, so use the capacity from NVIDIA's 6.0 surface demo.
        _physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(self.physics._physics_scene.GetPrim())
        _physx_scene_api.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1048576)
        # set camera prim view (zoomed in toward workspace center, configurable via env)
        _cam_eye = [float(x) for x in os.environ.get("STRETCH4_CAMERA_EYE", "0.0,3.15,2.7").split(",")]
        _cam_target = [float(x) for x in os.environ.get("STRETCH4_CAMERA_TARGET", "0.0,0.0,0.4").split(",")]
        set_camera_view(
            eye=_cam_eye,
            target=_cam_target,
            camera_prim_path="/OmniverseKit_Persp",
        )
        # set global light
        self.demo_light = rep.create.light(position=[0, 0, 0], light_type="dome")
        
        # set record flag
        self.record_flag = False
        # save recording data
        self.saving_data = []
        self.saving_data_replay = {
            "usd_path": None,
            "pos": None,
            "ori": None,
        }

    def reset(self):
        self.world.reset()

    def step(self):
        self.world.step(render=True)

    def stop(self):
        self.world.stop()

    def record(self, task_name: str, stage_index: int = 1):
        if self.record_flag == False:
            self.record_flag = True
            self.step_num = 0
            # create dir if not exist
            if not os.path.exists(f"Data/{task_name}/train_data/"):
                os.makedirs(f"Data/{task_name}/train_data/")
            if stage_index == 1:
                self.record_task_name = task_name
            self.stage_index = stage_index
            # add record_callback
            self.context.add_physics_callback("record_callback", self.record_callback)

    def stop_record(self):
        if self.record_flag == True:
            self.record_flag = False
            # remove record_callback
            self.context.remove_physics_callback("record_callback")
            # save record_data to target file
            self.saving_data_replay[f"stage_{self.stage_index}"] = np.array(
                self.saving_data
            )
            # clear record data
            self.saving_data = []

    def record_to_npz(self, env_change:bool=False):
        self.saving_data_replay["usd_path"] = self.garment.usd_path
        self.saving_data_replay["pos"] = self.position
        self.saving_data_replay["ori"] = self.orientation
        if env_change:
            self.saving_data_replay["env_dx"] = self.env_dx
            self.saving_data_replay["env_dy"] = self.env_dy
        # record_file_name = get_unique_filename(
        #     f"Data/{self.record_task_name}/train_data/data", ".pkl"
        # )
        # with open(record_file_name, "wb") as f:
        #     pickle.dump(self.saving_data_replay, f)
        record_file_name = get_unique_filename(
            f"Data/{self.record_task_name}/train_data/data", ".npz"
        )
        np.savez_compressed(record_file_name, **self.saving_data_replay)
        cprint(f"Record data saved to {record_file_name}", "green", "on_green")
        return record_file_name

    def replay(self, stage_index):
        # load data
        self.data = self.saving_data_replay[f"stage_{stage_index}"]
        # current timestep
        self.time_ptr = 0
        # whole length of data
        self.total_ticks = len(self.data)
        # add replay_callback
        self.context.add_physics_callback("replay_callback", self._replay_callback)

    def _replay_callback(
        self, step_size
    ):  # input parameter must be like (self, step_size)
        if self.time_ptr < self.total_ticks:
            self.replay_callback(self.data, self.time_ptr)
            self.time_ptr += 1
        else:
            self.context.remove_physics_callback("replay_callback")

    def record_callback(
        self, step_size
    ):  # input parameter must be like (self, step_size)
        """
        you can overwrite this function in specific env
        in order to satisfy personal record requirements
        """
        pass

    def replay_callback(self, data, time_ptr):
        """
        you can overwrite this function in specific env
        in order to satisfy personal replay requirements
        """
        pass

# if __name__=="__main__":
#     env = BaseEnv()
    
#     while simulation_app.is_running():
#         simulation_app.update()
        
#     simulation_app.close()
