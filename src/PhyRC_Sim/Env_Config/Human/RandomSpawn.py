"""Move the finished human and chair together without changing their geometry."""
import os

import numpy as np
from pxr import Gf, Usd, UsdGeom
from Env_Config.Randomization import spawn_randomization_enabled


def randomize_human_and_chair(stage, human_path):
    """Sample uniformly in a 10 cm disk and within +/-30 degrees of initial yaw.

    Call after pose baking, chair sizing and collider construction, before
    physics initialization or world-space contact caches are built. A shared
    outer rigid transform preserves all existing local transforms and points.
    HUMAN_SPAWN_SEED optionally reproduces a placement for debugging.
    """
    human = UsdGeom.Xformable(stage.GetPrimAtPath(human_path))
    pivot = human.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    if not spawn_randomization_enabled():
        # Leave the authored assembly untouched, including its transform stack.
        print('[Teleop] human/chair spawn: randomization disabled; original placement', flush=True)
        return {"offset_m": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "seed": None,
                "pivot_m": list(pivot), "randomized": False}
    seed = os.environ.get("HUMAN_SPAWN_SEED")
    seed = np.random.SeedSequence().entropy if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    radius = 0.10 * np.sqrt(rng.random())
    azimuth = rng.uniform(0.0, 2.0 * np.pi)
    offset = np.array([radius * np.cos(azimuth), radius * np.sin(azimuth), 0.0])
    yaw = float(rng.uniform(-30.0, 30.0))
    # USD uses row vectors: first subtract the shared pivot, then rotate,
    # then return to the pivot plus the sampled horizontal displacement.
    delta = (Gf.Matrix4d().SetTranslate(-pivot)
             * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw))
             * Gf.Matrix4d().SetTranslate(pivot + Gf.Vec3d(*offset)))
    paths = [human_path]
    if stage.GetPrimAtPath("/World/Chair"):
        UsdGeom.Xform.Define(stage, "/World/Chair")
        paths.append("/World/Chair")
    for path in paths:
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        previous = xf.GetOrderedXformOps()
        op = xf.AddTransformOp(opSuffix="randomSpawn")
        op.Set(delta)
        xf.SetXformOpOrder([op, *previous], xf.GetResetXformStack())
    print(f"[Teleop] human/chair spawn: offset=({offset[0]:+.5f}, {offset[1]:+.5f}) m, "
          f"radius={radius:.5f} m, yaw={yaw:.2f} deg, seed={seed}", flush=True)
    return {"offset_m": offset.tolist(), "yaw_deg": yaw, "seed": seed,
            "pivot_m": list(pivot), "randomized": True}


def placement_snapshot(stage):
    """Static scene transforms needed to validate a world-space checkpoint."""
    return {
        "placement_" + name.lower(): np.asarray(
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/" + name))
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
        for name in ("Human", "Chair") if stage.GetPrimAtPath("/World/" + name)
    }


def placement_matches(stage, saved):
    current = placement_snapshot(stage)
    keys = {key for key in saved if key.startswith("placement_")}
    return keys == current.keys() and all(
        saved[key].shape == value.shape
        and np.allclose(saved[key], value, rtol=0.0, atol=1e-8)
        for key, value in current.items())
