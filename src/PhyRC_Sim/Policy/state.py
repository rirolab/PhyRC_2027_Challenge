"""Read the running scene without changing physics, targets or control state.

Isaac-specific access is confined to snapshot(); numeric pose helpers can also
be used by offline dataset tools. Public poses use metres and wxyz quaternions.
"""
import numpy as np


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Expected a finite nonzero wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def relative_pose(base, pose):
    """Transform [xyz,wxyz] world pose into the base frame."""
    base, pose = np.asarray(base, dtype=float), np.asarray(pose, dtype=float)
    if base.shape != (7,) or pose.shape != (7,) or not np.isfinite(np.r_[base, pose]).all():
        raise ValueError('Expected two finite xyz+wxyz poses')
    rotation(base[3:])
    rotation(pose[3:])
    b = base[3:] / np.linalg.norm(base[3:])
    q = pose[3:] / np.linalg.norm(pose[3:])
    w, v = b[0], -b[1:]
    rel = np.r_[w*q[0]-v@q[1:], w*q[1:]+q[0]*v+np.cross(v, q[1:])]
    return np.r_[rotation(b).T @ (pose[:3]-base[:3]), rel]


def snapshot(module, cloths, rigs):
    """Return JSON metadata and full numeric arrays; all accesses are reads.

    Names, limits and counts are discovered from the live articulation/scene.
    Link poses come from PhysX, not potentially stale USD display transforms.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    stage = cloths[0].prim.GetStage()
    arrays, robots, garments = {}, [], []
    for i, rig in enumerate(rigs):
        robot = rig['robot']
        prefix = f'robot_{i}'
        names = list(robot.dof_names)
        q, qd = array(robot.get_joint_positions()).reshape(-1), array(robot.get_joint_velocities()).reshape(-1)
        limits = array(rig['ctrl'].get_joint_limits()).reshape(-1, 2)
        position, quaternion = robot.get_world_pose()
        base = np.r_[array(position).reshape(3), array(quaternion).reshape(4)]
        twist = np.r_[array(robot.get_linear_velocity()).reshape(3), array(robot.get_angular_velocity()).reshape(3)]
        transforms = array(robot._articulation_view._physics_view.get_link_transforms())[0]
        root = robot.prim_path.rsplit('/', 1)[0]
        joints = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
                  if p.IsA(UsdPhysics.Joint)}
        joint_metadata = []
        for j, name in enumerate(names):
            prim = joints.get(name)
            kind = prim.GetTypeName() if prim else 'unknown'
            unit = {'PhysicsRevoluteJoint': 'rad', 'PhysicsPrismaticJoint': 'm'}.get(kind, 'unknown')
            joint_metadata.append({'name': name, 'index': j, 'type': kind, 'unit': unit,
                                   'lower': float(limits[j, 0]), 'upper': float(limits[j, 1])})
        links = {}
        for role, index in [('grasp_center', rig['grasp_link_idx']),
                            ('fingertip_left', rig['fingertip_left_idx']),
                            ('fingertip_right', rig['fingertip_right_idx'])]:
            raw = transforms[index]
            pose = np.r_[raw[:3], raw[6], raw[3:6]]  # PhysX xyzw -> public wxyz.
            local = relative_pose(base, pose)
            links[role] = {'pose_world': pose.tolist(), 'pose_base': local.tolist(), 'physics_link_index': int(index)}
            arrays[f'{prefix}/{role}_pose_world'] = pose
            arrays[f'{prefix}/{role}_pose_base'] = local
        left, right = (np.array(links[key]['pose_world'][:3]) for key in ('fingertip_left', 'fingertip_right'))
        state = rig['state']
        targets = {name: float(state[name]) for name in ('lift', 'arm', 'yaw', 'pitch', 'roll', 'grip_pos',
                                                       'base_fwd', 'base_strafe', 'base_turn')}
        for name, values in [('joint_position', q), ('joint_velocity', qd), ('base_pose_world', base),
                             ('base_twist_world', twist), ('joint_limits', limits)]:
            arrays[f'{prefix}/{name}'] = values
        robots.append({'id': prefix, 'articulation_path': robot.prim_path, 'joint_count': len(names),
                       'joints': joint_metadata, 'joint_position': q.tolist(), 'joint_velocity': qd.tolist(),
                       'base_pose_world': base.tolist(), 'base_twist_world': twist.tolist(), 'links': links,
                       'fingertip_origin_distance_m': float(np.linalg.norm(left-right)),
                       'fingertip_midpoint_world': ((left+right)/2).tolist(),
                       'command_targets': targets, 'gripper_close_command': bool(state['gripper_closed']),
                       'grasp_attached': state.get('grabbed') is not None,
                       'native_attachment_present': bool(stage.GetPrimAtPath(root+'/ClothGraspAttachment')),
                       'wheel_joint_names': [name for name in names if name.startswith('wheel_')],
                       'gripper_camera_mount_path': root+'/gripper_camera_link',
                       'gripper_camera_mount_exists': bool(stage.GetPrimAtPath(root+'/gripper_camera_link'))})
    for i, cloth in enumerate(cloths):
        points = array(cloth.get_world_positions())[0]
        velocities = array(cloth.get_velocities())[0]
        mesh = UsdGeom.Mesh(cloth.prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError('Policy inspector currently expects triangular cloth topology')
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
        arrays[f'garment_{i}/position_world'] = points
        arrays[f'garment_{i}/velocity_world'] = velocities
        arrays[f'garment_{i}/triangles'] = faces
        garments.append({'id': f'garment_{i}', 'path': str(cloth.prim.GetPath()),
                         'vertices': len(points), 'triangles': len(faces),
                         'centroid_world': points.mean(axis=0).tolist(),
                         'aabb_min_world': points.min(axis=0).tolist(), 'aabb_max_world': points.max(axis=0).tolist(),
                         'finite': bool(np.isfinite(points).all() and np.isfinite(velocities).all())})
    static = {}
    for name in ('Human', 'Chair'):
        prim = stage.GetPrimAtPath('/World/'+name)
        if prim:
            matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            static[name.lower()] = {'world_matrix_usd_row_vectors': matrix.tolist(),
                                     'position_world': matrix[3, :3].tolist()}
    rates = {name: float(getattr(module, name)) for name in ('LIFT_RATE', 'ARM_RATE', 'WRIST_RATE',
             'GRIPPER_OPEN', 'GRIPPER_CLOSED', 'GRIPPER_RATE', 'BASE_LINEAR_RATE', 'BASE_ANGULAR_RATE',
             'BASE_LINEAR_ACCEL', 'BASE_ANGULAR_ACCEL')}
    result = {'schema_version': '0.2.0', 'units': 'm, rad, s; quaternions wxyz',
              'source': 'live PhysX state; static poses from USD; no camera images synthesized',
              'robots': robots, 'garments': garments, 'static': static, 'runtime_control_parameters': rates,
              'usd_camera_paths': [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)],
              'all_arrays_finite': bool(all(np.isfinite(v).all() for v in arrays.values()))}
    world = World.instance()
    result['simulation_time_s'] = float(world.current_time)
    result['physics_dt_s'] = float(world.get_physics_dt())
    result['render_dt_s'] = float(world.get_rendering_dt())
    return result, arrays
