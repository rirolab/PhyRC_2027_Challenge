from pathlib import Path
import json
import numpy as np

files = list(Path("/output/v2_sdl").glob("*/plans/plan_*_map.npz"))
if not files:
    raise SystemExit("No saved plan map")

f = max(files, key=lambda p: p.stat().st_mtime)
run = f.parent.parent

with np.load(f) as d:
    cost = d["base_cost_2d"]
    unknown = d["base_unknown_2d"]
    occupied = d["base_occupied_2d"]
    origin = d["origin"]
    res = float(d["resolution"])

report = json.loads((run / "report.json").read_text())
bases = np.asarray(report["handoff"]["base_pose_world"])[:, :2]

free_ij = np.argwhere(cost < 255)
bounds_max = origin + np.asarray(cost.shape) * res

print("RUN:", run.name)
print("MAP:", f.name)
print("BOUNDS:", origin.tolist(), bounds_max.tolist())
print("FREE:", len(free_ij))
print("UNKNOWN:", int(unknown.sum()))
print("OCCUPIED:", int(occupied.sum()))

for robot, xy in enumerate(bases):
    ij = np.floor((xy - origin) / res).astype(int)
    inside = np.all(ij >= 0) and np.all(ij < cost.shape)
    boundary = float(np.min(np.r_[xy-origin, bounds_max-xy]))
    print(f"\nR{robot} handoff XY:", xy.tolist())
    print("Grid cell:", ij.tolist())
    print("Distance to nearest map boundary:", round(boundary, 3), "m")

    if inside:
        ix, iy = ij
        print("Cost:", int(cost[ix, iy]))
        print("Unknown:", bool(unknown[ix, iy]))
        print("Occupied:", bool(occupied[ix, iy]))

        coarse = ij // 2
        block = cost[coarse[0]*2:coarse[0]*2+2,
                     coarse[1]*2:coarse[1]*2+2]
        print("A* coarse cell free:", bool(block.shape == (2,2) and np.all(block < 255)))
    else:
        print("OUTSIDE MAP")

    if len(free_ij):
        free_xy = origin + (free_ij + 0.5) * res
        nearest = float(np.linalg.norm(free_xy - xy, axis=1).min())
        print("Nearest traversable cell:", round(nearest, 3), "m")
