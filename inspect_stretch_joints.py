from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(
    "assets/custom/robots/stretch_4/stretch_4.usd"
)

root = stage.GetPrimAtPath("/stretch/joints")

for prim in Usd.PrimRange(root):
    if not (
        prim.IsA(UsdPhysics.PrismaticJoint)
        or prim.IsA(UsdPhysics.RevoluteJoint)
    ):
        continue

    joint = UsdPhysics.Joint(prim)

    body0 = joint.GetBody0Rel().GetTargets()
    body1 = joint.GetBody1Rel().GetTargets()

    if prim.IsA(UsdPhysics.PrismaticJoint):
        typed = UsdPhysics.PrismaticJoint(prim)
        kind = "PRISMATIC"
    else:
        typed = UsdPhysics.RevoluteJoint(prim)
        kind = "REVOLUTE"

    lower = typed.GetLowerLimitAttr().Get()
    upper = typed.GetUpperLimitAttr().Get()
    axis = typed.GetAxisAttr().Get()

    print(f"\n{prim.GetName()} [{kind}]")
    print("  body0:", [str(p) for p in body0])
    print("  body1:", [str(p) for p in body1])
    print("  axis:", axis)
    print("  lower:", lower)
    print("  upper:", upper)
    print("  localPos0:", joint.GetLocalPos0Attr().Get())
    print("  localPos1:", joint.GetLocalPos1Attr().Get())
