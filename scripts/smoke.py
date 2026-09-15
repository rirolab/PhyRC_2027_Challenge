"""Exercise the real teleop main loop, controls and checkpoint API on the GPU.

Uses an isolated state directory, never a user's F1-F5 slots. This is a short
installation smoke test, not a dressing-retention or high-load certification.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

output = Path('/output/verification')
output.mkdir(parents=True, exist_ok=True)
os.environ['STRETCH4_STATE_DIR'] = tempfile.mkdtemp(prefix='smoke_states_', dir=output)
sys.path.insert(0, '/workspace/PhyRC_Sim')
from policy_cli import install_failure_handler
install_failure_handler()
import numpy as np
import Env_StandAlone.Teleop_TShirt_Stretch4_Env as M
from pxr import Usd, UsdGeom
import omni.usd

report = {'passed': False, 'scope': 'real main-loop startup, unloaded robot motion, gripper toggle, save/load'}
context = {'stop': False}
original_save = M.save_state_slot
original_drive = M.drive_robot
original_running = M.simulation_app.is_running


def array(value):
    return np.asarray(M._to_np(value)).copy()


def geometry_hash(value):
    rounded = np.round(np.asarray(value, dtype=np.float64), 6)
    return hashlib.sha256(rounded.astype('<f8').tobytes()).hexdigest()


def observed_save(key, cloths, rigs):
    result = original_save(key, cloths, rigs)
    if key == 'F1' and 'cloths' not in context:
        assert result, 'Startup checkpoint could not be saved'
        context.update(cloths=cloths, rigs=rigs)
        stage = omni.usd.get_context().get_stage()
        mesh = UsdGeom.Mesh(cloths[0].prim)
        material_path = str(cloths[0].prim.GetPath()).rsplit('/', 2)[0]
        materials = {}
        heads = []
        for prim in stage.Traverse():
            values = {a.GetName(): a.Get() for a in prim.GetAttributes()
                      if a.GetName().startswith(('omniphysics:', 'physxDeformableMaterial:'))
                      and a.Get() is not None}
            if values and prim.GetTypeName() == 'Material':
                materials[str(prim.GetPath())] = values
            info = prim.GetAttribute('phyrc:roundedHeadInfo')
            if info and info.Get():
                heads.append({'info': json.loads(info.Get()),
                              'points_sha256': geometry_hash(UsdGeom.Mesh(prim).GetPointsAttr().Get())})
        body = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Human/CollisionBody'))
        assert body, 'Rounded human collision mesh is missing'
        report['scene'] = {
            'garments': len(cloths), 'robots': len(rigs),
            'vertices': len(mesh.GetPointsAttr().Get()),
            'triangles': len(mesh.GetFaceVertexCountsAttr().Get()),
            'rest_points_sha256': geometry_hash(mesh.GetPointsAttr().Get()),
            'topology_sha256': geometry_hash(mesh.GetFaceVertexIndicesAttr().Get()),
            'collision_points_sha256': geometry_hash(body.GetPointsAttr().Get()),
            'solver_iterations': cloths[0].prim.GetAttribute('physxDeformableBody:solverPositionIterationCount').Get(),
            'materials': materials, 'rounded_heads': heads,
            'robot_initial_positions': [array(r['robot'].get_world_pose()[0]).tolist() for r in rigs],
            'human_transform': np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath('/World/Human'))
                                         .ComputeLocalToWorldTransform(Usd.TimeCode.Default())).tolist(),
        }
        context['initial_joints'] = [array(r['robot'].get_joint_positions()) for r in rigs]
        context['faces'] = [M.garment_face_labels(c.prim, array(c.get_world_positions())[0]) for c in cloths]
        # Current distributed V-neck asset after the existing midpoint refinement.
        # These are checks only; do not change scene geometry/physics to fit a test.
        for field, expected in {'vertices': 15878, 'triangles': 31344,
                                'solver_iterations': 64}.items():
            actual = report['scene'][field]
            assert actual == expected, f'{field}: expected {expected}, got {actual}'
        assert heads, 'Rounded visual head metadata is missing'
        print('PHYRC-SMOKE: full main-loop initialization reached', flush=True)
    return result


def observed_drive(rig, keymap, held, cloths, dt, frame):
    requested = {keymap['lift_pos']} if frame < 12 else set()
    original_drive(rig, keymap, requested, cloths, dt, frame)
    if rig is not context['rigs'][0]:
        return
    if frame == 13:
        toggle = M.make_gripper_toggle(rig, 'smoke', cloths, context['faces'])
        toggle()
        assert rig['state']['gripper_closed'], 'Gripper close did not reach control state'
        report['grasp_attached_in_initial_pose'] = rig['state'].get('grabbed') is not None
        toggle()
        assert not rig['state']['gripper_closed'], 'Gripper release did not reach control state'
        report['gripper_toggle'] = True
    if frame == 24:
        rigs = context['rigs']
        assert original_save('F2', cloths, rigs), 'Save failed'
        before = [array(c.get_world_positions()) for c in cloths]
        for cloth in cloths:
            positions = cloth.get_world_positions().clone()
            positions[..., 0] += 0.001
            cloth.set_world_positions(positions)
        assert M.load_state_slot('F2', cloths, rigs), 'Load failed'
        error = max(float(np.max(np.abs(array(c.get_world_positions()) - p)))
                    for c, p in zip(cloths, before))
        report['save_load_max_position_error_m'] = error
        assert error < 1e-5, f'Checkpoint position mismatch: {error}'
    if frame >= 36:
        movement = [float(np.max(np.abs(array(r['robot'].get_joint_positions()) - p)))
                    for r, p in zip(context['rigs'], context['initial_joints'])]
        assert all(value > 1e-4 for value in movement), f'Robots did not move: {movement}'
        assert all(np.isfinite(array(c.get_world_positions())).all() for c in cloths)
        report.update(passed=True, simulated_control_frames=frame + 1,
                      max_joint_displacement=movement, finite_cloth=True)
        (output / 'smoke.json').write_text(json.dumps(report, indent=2, default=str))
        print('PHYRC-SMOKE-PASS ' + json.dumps(report, default=str), flush=True)
        context['stop'] = True


M.save_state_slot = observed_save
M.drive_robot = observed_drive
M.simulation_app.is_running = lambda: not context['stop'] and original_running()
try:
    M.main()
except BaseException as error:
    report['error'] = repr(error)
    (output / 'smoke.json').write_text(json.dumps(report, indent=2, default=str))
    raise
if not report['passed']:
    raise SystemExit('Simulation exited before the smoke test completed')
