"""Let the cloth's low friction control contact with the human colliders."""
import os
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade


def configure_human_contact(stage, root_path):
    # PhysX 110 reads rigid/deformable combine mode from the RIGID material.
    # With the default average rule, zero-friction cloth still feels half of
    # the body's friction. min lets it slide without changing normal contact.
    mode = os.environ.get('STRETCH4_HUMAN_FRICTION_COMBINE', 'min')
    if mode not in ('min', 'average', 'multiply', 'max'):
        raise ValueError(f'invalid human friction combine mode: {mode}')
    colliders = [p for p in Usd.PrimRange(stage.GetPrimAtPath(root_path),
                                         Usd.TraverseInstanceProxies())
                 if p.IsA(UsdGeom.Gprim) and p.HasAPI(UsdPhysics.CollisionAPI)
                 and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    for index, collider in enumerate(colliders):
        # Retain the collider's previous rigid coefficients; change only how
        # its friction combines with the contacting cloth. No shared material
        # (e.g. a floor or robot material) is edited.
        bound, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
        previous = UsdPhysics.MaterialAPI(bound.GetPrim()) if bound else None
        values = {}
        for name, fallback in (('StaticFriction', .5), ('DynamicFriction', .5),
                               ('Restitution', 0.)):
            value = getattr(previous, 'Get' + name + 'Attr')().Get() if previous else None
            values[name] = fallback if value is None else value
        material = UsdShade.Material.Define(stage, f'{root_path}/ClothContactMaterial_{index}')
        rigid = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        for name, value in values.items():
            getattr(rigid, 'Create' + name + 'Attr')(value)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr(mode)
        UsdShade.MaterialBindingAPI.Apply(collider).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, 'physics')
    print(f'[Teleop] cloth/human friction combine={mode}: {len(colliders)} colliders', flush=True)
