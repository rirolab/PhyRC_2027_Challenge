from pathlib import Path
from collections import Counter
from pxr import Usd, UsdPhysics, UsdGeom

path = Path("assets/custom/robots/stretch_4/stretch_4.usd").resolve()
stage = Usd.Stage.Open(str(path))

if not stage:
    raise RuntimeError(f"Cannot open USD: {path}")

print("USD:", path)
print("Up axis:", UsdGeom.GetStageUpAxis(stage))
print("Meters per unit:", UsdGeom.GetStageMetersPerUnit(stage))

counts = Counter()
colliders = []
joints = []
references = []
instances = []

for prim in stage.Traverse():
    counts[prim.GetTypeName()] += 1

    if prim.IsInstance():
        instances.append(str(prim.GetPath()))

    if prim.HasAuthoredReferences():
        references.append(str(prim.GetPath()))

    if prim.HasAPI(UsdPhysics.CollisionAPI):
        attr = UsdPhysics.CollisionAPI(
            prim
        ).GetCollisionEnabledAttr()

        enabled = attr.Get()
        colliders.append((
            str(prim.GetPath()),
            prim.GetTypeName(),
            enabled
        ))

    if prim.IsA(UsdPhysics.Joint):
        joints.append((
            str(prim.GetPath()),
            prim.GetTypeName()
        ))

print("\n=== PRIM TYPES ===")
for key, count in sorted(counts.items()):
    print(f"{key}: {count}")

print("\n=== COLLISION SHAPES ===")
for path, typ, enabled in colliders:
    print(f"{typ:12s} enabled={enabled} {path}")

print("\n=== JOINTS ===")
for path, typ in joints:
    print(typ, path)

print("\n=== REFERENCES ===")
for path in references:
    print(path)

print("\n=== INSTANCE ROOTS ===")
for path in instances:
    print(path)

print("\n=== SUMMARY ===")
print("Collision prims:", len(colliders))
print("Enabled collision prims:",
      sum(enabled is not False for _, _, enabled in colliders))
print("Joint prims:", len(joints))
print("Instance roots:", len(instances))
