"""Zero scene friction, then configure selected contact materials before cooking."""
import os
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade


def zero_scene_friction(stage):
    # Material coefficients are stage-local overrides, never source USD edits.
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            api = UsdPhysics.MaterialAPI(prim)
            api.CreateStaticFrictionAttr(0.0)
            api.CreateDynamicFrictionAttr(0.0)
            PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr('min')
        for name in ('omniphysics:staticFriction', 'omniphysics:dynamicFriction'):
            attr = prim.GetAttribute(name)
            if attr:
                attr.Set(0.0)
    count = 0
    colliders = [p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                 if p.HasAPI(UsdPhysics.CollisionAPI)
                 and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    for collider in colliders:
        material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
        if material and material.GetPrim().HasAPI('OmniPhysicsDeformableMaterialAPI'):
            continue
        api = UsdPhysics.MaterialAPI(material.GetPrim()) if material else None
        if api and api.GetStaticFrictionAttr().Get() == 0 and api.GetDynamicFrictionAttr().Get() == 0:
            count += 1
            continue
        # Shapes without an explicit physics material otherwise inherit the
        # engine's nonzero default. Bind at the editable instance root for
        # robot instance proxies, preserving all previous material attributes.
        target = collider
        while target.IsInstanceProxy():
            target = target.GetParent()
        path = f'/World/ZeroFrictionMaterials/material_{count}'
        replacement = UsdShade.Material.Define(stage, path)
        rigid = UsdPhysics.MaterialAPI.Apply(replacement.GetPrim())
        if material:
            for attr in material.GetPrim().GetAttributes():
                if attr.GetName().startswith(('physics:', 'physxMaterial:')) and attr.Get() is not None:
                    replacement.GetPrim().CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
        rigid.CreateStaticFrictionAttr(0.0)
        rigid.CreateDynamicFrictionAttr(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(replacement.GetPrim()).CreateFrictionCombineModeAttr('min')
        UsdShade.MaterialBindingAPI.Apply(target).Bind(
            replacement, UsdShade.Tokens.strongerThanDescendants, 'physics')
        count += 1
    print(f'[Zero friction] {count} rigid colliders plus deformable materials: static=dynamic=0', flush=True)

    from .HumanClothFriction import configure_human_cloth_friction
    configure_human_cloth_friction(stage, os.environ.get('STRETCH4_HUMAN_CONTACT_FRICTION', '0'))

    from .GripperClothFriction import configure_gripper_cloth_friction
    configure_gripper_cloth_friction(stage, os.environ.get('STRETCH4_GRIPPER_CONTACT_FRICTION', '0.5'))

    from .TableClothFriction import configure_table_cloth_friction
    configure_table_cloth_friction(stage, os.environ.get('STRETCH4_TABLE_CONTACT_FRICTION', '0.5'))
