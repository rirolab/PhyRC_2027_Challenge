"""Hide separate robot collision geometry from rendering, preserving physics."""
from pxr import Usd, UsdGeom, UsdPhysics


def hide_robot_collision_visuals(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    hidden = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        # Only the explicitly separate collision subtree of a link which also
        # has a visual subtree. Never hide a dual-purpose visible collider.
        if prim.GetName() != 'collisions' or prim.IsInstanceProxy():
            continue
        if not prim.GetParent().GetChild('visuals').IsValid():
            continue
        shapes = [p for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                  if p.HasAPI(UsdPhysics.CollisionAPI)]
        enabled = [UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in shapes]
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        assert enabled == [UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in shapes]
        hidden += len(shapes)
    print(f'[Render] {root_path}: hidden {hidden} separate collision shapes; physics unchanged', flush=True)
    return hidden
