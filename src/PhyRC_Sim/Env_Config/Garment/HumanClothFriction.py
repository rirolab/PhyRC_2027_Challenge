"""Apply an explicit cloth/human coefficient after the scene's zero-friction pass."""
import math
from pxr import PhysxSchema, UsdPhysics, UsdShade


def configure_human_cloth_friction(stage, coefficient):
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('cloth/human friction must be finite and nonnegative')
    human = stage.GetPrimAtPath('/World/Human')
    if coefficient == 0 or not human:
        return
    material = UsdShade.Material.Define(stage, '/World/HumanClothContactMaterial')
    rigid = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    rigid.CreateStaticFrictionAttr(coefficient)
    rigid.CreateDynamicFrictionAttr(coefficient)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr('min')
    # The referenced human root already has a stronger-than-descendants binding.
    # A child-only override leaves the effective collider material at zero.
    UsdShade.MaterialBindingAPI.Apply(human).Bind(
        material, UsdShade.Tokens.strongerThanDescendants, 'physics')
    checked = 0
    for path in ('/World/Human/CollisionBody', '/World/Human/HandSphere_left',
                 '/World/Human/HandSphere_right'):
        collider = stage.GetPrimAtPath(path)
        if not collider or not collider.HasAPI(UsdPhysics.CollisionAPI):
            continue
        bound, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
        if not bound or bound.GetPath() != material.GetPath():
            raise RuntimeError(f'cloth/human friction binding was overridden: {path}')
        checked += 1
    for prim in stage.Traverse():
        for name in ('omniphysics:staticFriction', 'omniphysics:dynamicFriction'):
            attr = prim.GetAttribute(name)
            if attr:
                attr.Set(coefficient)
    print(f'[Cloth/human friction] static=dynamic={coefficient:g}, '
          f'combine=min, {checked} effective human collider bindings verified', flush=True)
