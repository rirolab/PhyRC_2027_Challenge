import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/project/participant")
from strategy_core import CoupledBaseAStar

files = list(Path("/output/v2_sdl").glob(
    "*/perception_review/grounded_evidence_map.npz"
))
if not files:
    raise SystemExit("Run diagnose_ground_evidence.py first")

p = max(files, key=lambda x: x.stat().st_mtime)
print("SOURCE:", p)

with np.load(p) as data:
    origin = data["origin"]
    resolution = float(data["resolution"])
    xy = data["base_xy"]

    for map_name in ("strict_cost_2d", "broad_review_cost_2d"):
        print("\nMAP:", map_name)

        for step in (2, 1):
            planner = CoupledBaseAStar(
                data[map_name], origin, resolution,
                step_cells=step,
                min_base_separation=0.43
            )

            a = planner.cell(xy[0])
            b = planner.cell(xy[1])
            sep = np.linalg.norm(
                (np.array(a) - np.array(b)) * planner.res
            )
            pair = (*a, *b)

            print(
                f"step={step}",
                f"distance={sep:.3f}m",
                f"R0_FREE={planner._valid_one(a)}",
                f"R1_FREE={planner._valid_one(b)}",
                f"PAIR_VALID={planner._valid(pair)}"
            )
