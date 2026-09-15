"""Restore finger contact friction while leaving the zero-friction cloth intact."""
import math

from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade


def configure_gripper_cloth_friction(stage, coefficient=0.5):
    """Bind only finger/fingertip shapes, including instance-proxy colliders.

    Rigid/FEM contact uses the rigid material's combine rule. max(mu, 0)
    gives the requested grip friction without adding cloth self-friction or
    cloth/human friction. Zero restores min/0, the previous scene behavior.
    Native grasp attachments are independent and are not modified here.
    """
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('gripper/cloth friction must be finite and nonnegative')
    mode = 'max' if coefficient else 'min'
    selected = []
    targets = {}
    for root in ('/World/Stretch4', '/World/Stretch4_2'):
        robot = stage.GetPrimAtPath(root)
        if not robot:
            continue
        for collider in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
            relative = str(collider.GetPath())[len(root) + 1:]
            link = relative.split('/')[0]
            if link not in ('gripper_finger_left_link', 'gripper_finger_right_link',
                            'gripper_fingertip_left_link', 'gripper_fingertip_right_link'):
                continue
            if (not collider.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False):
                continue
            target = collider
            while target.IsInstanceProxy():
                target = target.GetParent()
            if not target.GetPath().HasPrefix(robot.GetPath().AppendChild(link)):
                raise RuntimeError(f'Cannot isolate finger material at {collider.GetPath()}')
            selected.append(collider)
            targets[str(target.GetPath())] = target

    for index, target in enumerate(targets.values()):
        previous, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial('physics')
        material = UsdShade.Material.Define(stage, f'/World/GripperContactMaterials/material_{index}')
        # Keep existing restitution/compliance attributes when replacing a binding.
        if previous and previous.GetPath() != material.GetPath():
            for attr in previous.GetPrim().GetAttributes():
                if attr.GetName().startswith(('physics:', 'physxMaterial:')) and attr.Get() is not None:
                    material.GetPrim().CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
        rigid = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        rigid.CreateStaticFrictionAttr(coefficient)
        rigid.CreateDynamicFrictionAttr(coefficient)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr(mode)
        UsdShade.MaterialBindingAPI.Apply(target).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, 'physics')

    for collider in selected:
        material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
        rigid = UsdPhysics.MaterialAPI(material.GetPrim())
        if (not math.isclose(rigid.GetDynamicFrictionAttr().Get(), coefficient, abs_tol=1e-7)
                or PhysxSchema.PhysxMaterialAPI(material.GetPrim()).GetFrictionCombineModeAttr().Get() != mode):
            raise RuntimeError(f'Gripper material binding overridden: {collider.GetPath()}')
    report = {'coefficient': coefficient, 'combine': mode,
              'colliders': [str(p.GetPath()) for p in selected]}
    print(f'[Gripper/cloth friction] coefficient={coefficient:g}, combine={mode}, '
          f'{len(selected)} effective finger collider bindings verified', flush=True)
    return report
