"""Restore table/box contact friction while keeping human contact frictionless."""
import math
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade


def configure_table_cloth_friction(stage, coefficient=0.5):
    """Bind table/box colliders with restored friction.

    Rigid/FEM contact uses the rigid material's combine rule. max(mu, 0)
    gives the requested table friction so the cloth does not slip like ice,
    without affecting cloth/human friction. Zero restores min/0.
    """
    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('table/cloth friction must be finite and nonnegative')
    mode = 'max' if coefficient else 'min'
    selected = []
    for root_path in ('/World/garment_table_0', '/World/garment_table_1',
                      '/World/garment_table_2', '/World/garment_table_3', '/World/Table'):
        table = stage.GetPrimAtPath(root_path)
        if not table or not table.IsValid():
            continue
        for prim in Usd.PrimRange(table, Usd.TraverseInstanceProxies()):
            if (prim.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False):
                selected.append(prim)

    for index, target in enumerate(selected):
        previous, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial('physics')
        material = UsdShade.Material.Define(stage, f'/World/TableContactMaterials/material_{index}')
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
            raise RuntimeError(f'Table material binding overridden: {collider.GetPath()}')

    report = {'coefficient': coefficient, 'combine': mode,
              'colliders': [str(p.GetPath()) for p in selected]}
    print(f'[Table/cloth friction] coefficient={coefficient:g}, combine={mode}, '
          f'{len(selected)} table collider bindings verified', flush=True)
    return report
