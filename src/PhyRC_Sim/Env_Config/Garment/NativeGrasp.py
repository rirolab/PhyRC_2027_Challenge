"""Isaac 6 vertex attachments for a real robot's garment grasp."""
import numpy as np
import torch
from pxr import Gf


def clear_native_grasp(rig, stage):
    root = rig['robot'].prim_path.rsplit('/', 1)[0]
    path = root + '/ClothGraspAttachment'
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
    rig['state'].pop('_native_grab', None)


def current_grasp_offsets(rig):
    """World offsets for saving a grasp after the wrist has rotated."""
    state = rig['state']
    grabbed = state['grabbed']
    offsets = grabbed[2]
    if state.get('_native_grab') is not grabbed or '_native_local_offsets' not in state:
        return offsets
    tf = rig['robot']._articulation_view._physics_view.get_link_transforms()
    tf = tf.detach().cpu().numpy()[0, rig['grasp_link_idx']]
    rotation = Gf.Rotation(Gf.Quatd(float(tf[6]), Gf.Vec3d(*map(float, tf[3:6]))))
    world = np.array([rotation.TransformDir(Gf.Vec3d(*map(float, p)))
                      for p in state['_native_local_offsets']], dtype=np.float32)
    return torch.as_tensor(world, dtype=offsets.dtype, device=offsets.device)


def update_native_grasp(rig, cloths, anchor_count=12):
    """Create/remove the attachment only when the selected grasp changes.

    PhysX solves these vertices together with FEM and contact. No nodal pose
    or correction velocity is written every step by the grasp controller.
    Synthetic test grips without a robot keep using the swept legacy adapter.
    """
    if 'robot' not in rig:
        return False
    state = rig['state']
    state['_native_enabled'] = True
    grabbed = state.get('grabbed')
    active = state.get('_native_grab')
    if active is grabbed:
        return True
    # The mesh and grasp_center_link are in the same stage.
    stage = cloths[0].prim.GetStage()
    root = rig['robot'].prim_path.rsplit('/', 1)[0]
    path = root + '/ClothGraspAttachment'
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
    if state.pop('_native_wait_for_fk', False) and grabbed is not None:
        # Restoring root/joint tensors updates child link FK on the next solve.
        # Delay attachment-frame conversion for that one substep.
        state.pop('_native_grab', None)
        return True
    state['_native_grab'] = grabbed
    if grabbed is None:
        return True
    ci, idx, offsets = grabbed
    count = min(max(int(anchor_count), 1), len(idx))
    mask = torch.zeros(len(idx), dtype=torch.bool, device=idx.device)
    mask[torch.argsort(torch.linalg.norm(offsets, dim=1))[:count]] = True
    state['_grab_anchor_mask'] = mask
    state['_grab_anchor_idx'] = idx.clone()
    ids = idx[mask].detach().cpu().numpy().astype(np.int32).tolist()
    link_path = root + '/grasp_center_link'
    link = stage.GetPrimAtPath(link_path)
    if not link:
        raise RuntimeError(f'Missing native grasp link: {link_path}')
    tf = rig['robot']._articulation_view._physics_view.get_link_transforms()
    tf = tf.detach().cpu().numpy()[0, rig['grasp_link_idx']]
    rotation = Gf.Rotation(Gf.Quatd(float(tf[6]), Gf.Vec3d(*map(float, tf[3:6]))).GetInverse())
    # Offsets are stored in world axes in old checkpoints. Convert to the
    # actual physics link frame; USD transforms may lag a restored tensor pose.
    local_all = np.array([rotation.TransformDir(Gf.Vec3d(*map(float, p)))
                          for p in offsets.detach().cpu().numpy()], dtype=np.float32)
    state['_native_local_offsets'] = local_all
    local = [Gf.Vec3f(*map(float, p)) for p in local_all[mask.detach().cpu().numpy()]]
    attachment = stage.DefinePrim(path, 'OmniPhysicsVtxXformAttachment')
    attachment.GetRelationship('omniphysics:src0').SetTargets([cloths[ci].prim_path])
    attachment.GetRelationship('omniphysics:src1').SetTargets([link_path])
    attachment.GetAttribute('omniphysics:vtxIndicesSrc0').Set(ids)
    attachment.GetAttribute('omniphysics:localPositionsSrc1').Set(local)
    print(f'[Teleop] native grasp: {count} vertices on {cloths[ci].prim_path}', flush=True)
    return True
