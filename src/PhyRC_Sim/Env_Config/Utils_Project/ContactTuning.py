"""Author contact settings on actual collision shapes, including USD instances."""
import os
from pxr import PhysxSchema, Usd, UsdPhysics


def tune_robot_contacts(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    # Proxy prims are read-only. Uninstance only collision-containing subtrees
    # before writing shape attributes; authoring on the link Xform is not enough.
    while True:
        instances = set()
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)) and prim.IsInstanceProxy():
                ancestor = prim
                while ancestor.IsInstanceProxy():
                    ancestor = ancestor.GetParent()
                instances.add(ancestor.GetPath())
        if not instances:
            break
        for path in instances:
            stage.GetPrimAtPath(path).SetInstanceable(False)

    contact = float(os.environ.get("STRETCH4_ROBOT_CONTACT_OFFSET", "0.006"))
    rest = float(os.environ.get("STRETCH4_ROBOT_REST_OFFSET", "0.001"))
    depen = float(os.environ.get("STRETCH4_MAX_DEPENETRATION_VELOCITY", "3.0"))
    if not 0.0 <= rest < contact or depen <= 0.0:
        raise ValueError("robot contact must exceed nonnegative rest offset; depenetration speed must be positive")
    bodies = shapes = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            api.CreateMaxDepenetrationVelocityAttr().Set(depen)
            api.CreateEnableSpeculativeCCDAttr().Set(True)
            bodies += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            # Tune the manipulator. Wheel contact should retain its authored
            # rolling geometry and ground clearance.
            link = str(prim.GetPath()).removeprefix(root_path + "/").split("/")[0]
            if link.startswith(("gripper_", "arm_", "wrist", "lift_")):
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr().Set(contact)
                api.CreateRestOffsetAttr().Set(rest)
                shapes += 1
    print(f"[Contact] {root_path}: {bodies} bodies depenetration<={depen:g}m/s, "
          f"{shapes} manipulator shapes contact/rest={contact:g}/{rest:g}m", flush=True)
