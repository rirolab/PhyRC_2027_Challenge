"""Synchronous RGB/optical-Z sensors; rendering never advances physics."""
import numpy as np
from .state import array, rotation


def pose_matrix(raw):
    """PhysX xyz+xyzw to column-vector local-to-world matrix."""
    out = np.eye(4)
    out[:3, :3] = rotation(np.r_[raw[6], raw[3:6]])
    out[:3, 3] = raw[:3]
    return out


def look_at(eye, target, up):
    """USD camera basis: right, up, back (optical forward is -Z)."""
    back = np.asarray(eye) - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    out = np.eye(4)
    out[:3, :3] = np.column_stack((right, np.cross(back, right), back))
    out[:3, 3] = eye
    return out


class RGBDSensors:
    def __init__(self, stage, world, rigs, config):
        from pxr import UsdGeom, Gf
        import omni.replicator.core as rep
        rep.orchestrator.set_capture_on_play(False)
        self.stage, self.world, self.rigs = stage, world, rigs
        dims = config['default_dimensions']
        self.height, self.width = dims['height'], dims['width']
        self.items = []
        for spec in config['cameras']:
            camera = UsdGeom.Camera.Define(stage, '/World/PolicySensors/' + spec['id'])
            camera.CreateProjectionAttr('perspective')
            aperture = 20.955
            focal = aperture / (2 * np.tan(np.deg2rad(spec['horizontal_fov_deg']) / 2))
            camera.CreateHorizontalApertureAttr(aperture)
            camera.CreateVerticalApertureAttr(aperture * self.height / self.width)
            camera.CreateFocalLengthAttr(float(focal))
            camera.CreateClippingRangeAttr(Gf.Vec2f(*spec['clip_m']))
            item = {'spec': spec, 'op': camera.AddTransformOp(), 'camera': camera}
            if spec.get('mount') in ('world', 'human_spawn'):
                item['fixed_pose'] = look_at(np.array(spec['eye_world_m'], dtype=float),
                                            spec['target_world_m'], [0, 0, 1])
            else:
                index = int(spec['id'].split('_')[1])
                rig = rigs[index]
                link_index = rig['robot']._articulation_view.get_link_index(spec['mount_role'])
                if link_index is None or link_index < 0:
                    raise ValueError('Missing camera mount: ' + spec['mount_role'])
                item.update(robot_index=index, link_index=link_index)
                # Fixed extrinsic expressed in the selected link frame.
                # Wrist and head links have different bases; config provides
                # their respective forward/up directions explicitly.
                # Offset and aim are explicit configuration, never recomputed
                # from a moving target or a noisy USD display transform.
                item['mount_from_usd_camera'] = look_at(
                    np.array(spec['eye_mount_m'], dtype=float), spec['target_mount_m'], spec['up_mount'])
            product = rep.create.render_product(str(camera.GetPath()), (self.width, self.height))
            item.update(product=product, rgb=rep.AnnotatorRegistry.get_annotator('rgb'),
                        depth=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane'))
            item['rgb'].attach([product])
            item['depth'].attach([product])
            self.items.append(item)
        self.reset_episode()

    def reset_episode(self):
        """Place overview once using the same rigid spawn delta as the mannequin.

        Use only the outer randomSpawn op, not the human's mesh scale/baked
        pose rotation. Saved slots retain the equivalent delta as metadata.
        """
        from pxr import UsdGeom
        for item in self.items:
            spec = item['spec']
            if spec.get('mount') != 'human_spawn':
                continue
            prim = self.stage.GetPrimAtPath(spec['spawn_prim_path'])
            if not prim:
                raise ValueError('Missing camera spawn reference: ' + spec['spawn_prim_path'])
            delta = np.eye(4)
            restored = prim.GetAttribute('phyrc:cameraSpawnDelta')
            if restored and restored.HasAuthoredValueOpinion():
                delta = np.asarray(restored.Get(), dtype=float).T
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpName() == 'xformOp:transform:randomSpawn':
                    delta = np.asarray(op.Get(), dtype=float).T
                    break
            reference = look_at(np.array(spec['eye_world_m'], dtype=float),
                                spec['target_world_m'], [0, 0, 1])
            item['fixed_pose'] = delta @ reference
            if not np.isfinite(item['fixed_pose']).all():
                raise ValueError('Nonfinite camera spawn transform')

    def capture(self):
        from pxr import Gf
        timestamp = float(self.world.current_time)
        metadata = []
        for item in self.items:
            spec = item['spec']
            if 'fixed_pose' in item:
                matrix = item['fixed_pose']
            else:
                rig = self.rigs[item['robot_index']]
                raw = array(rig['robot']._articulation_view._physics_view.get_link_transforms())[0, item['link_index']]
                matrix = pose_matrix(raw) @ item['mount_from_usd_camera']
            item['op'].Set(Gf.Matrix4d(matrix.T.tolist()))
            focal_px = self.width / (2 * np.tan(np.deg2rad(spec['horizontal_fov_deg']) / 2))
            optical = matrix @ np.diag([1, -1, -1, 1])
            metadata.append({'id': spec['id'], 'timestamp_simulation_s': timestamp,
                             'resolution_wh': [self.width, self.height], 'clip_m': spec['clip_m'],
                             'intrinsics': [[focal_px, 0, self.width / 2], [0, focal_px, self.height / 2], [0, 0, 1]],
                             'world_from_optical_column_vectors': optical.tolist(),
                             'mount_from_usd_camera_column_vectors': item.get('mount_from_usd_camera', np.eye(4)).tolist()})
        # Sync physics -> render scene, then explicitly await THIS capture.
        # Plain app/render updates can return the preceding annotator frame.
        import omni.replicator.core as rep
        self.world.render()
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0,
                              pause_timeline=False, wait_for_render=True)
        if abs(float(self.world.current_time) - timestamp) > 1e-9:
            raise RuntimeError('Sensor rendering advanced the simulation clock')
        rgbs, depths, masks = [], [], []
        for item in self.items:
            rgb = np.asarray(item['rgb'].get_data())
            depth = np.asarray(item['depth'].get_data(), dtype=np.float32)
            if rgb.shape != (self.height, self.width, 4) or depth.shape != (self.height, self.width):
                raise RuntimeError(f'Sensor product not ready: RGB {rgb.shape}, depth {depth.shape}')
            near, far = item['spec']['clip_m']
            valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
            rgbs.append(rgb[:, :, :3].copy())
            depths.append(np.where(valid, depth, 0)[..., None])
            masks.append(valid[..., None])
        return {'rgb': np.stack(rgbs), 'depth': np.stack(depths), 'depth_valid': np.stack(masks)}, metadata

    def close(self):
        for item in self.items:
            for name in ('rgb', 'depth'):
                item[name].detach([item['product']])
            item['product'].destroy()
        self.items.clear()
