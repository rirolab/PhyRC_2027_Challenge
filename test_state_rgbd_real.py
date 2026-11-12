import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from phase2_reorientation.perception import MultiHSVRangeSegmenter
from phase2_reorientation.state_estimation import estimate_garment_state

# Same prompt construction as temporal gate v1
from phase2_reorientation_validation.tools.perception_temporal_gate_v1 import prompt

FRAMES = [1700, 1730, 1760, 2350, 2450]
CAMERAS = ["robot_0_head", "robot_1_head"]

HDF5 = "research_data/perception_validation/machine_01/policy.hdf5"
OUTPUT = Path("research_output/state_estimation_v1")
OUTPUT.mkdir(parents=True, exist_ok=True)

CALIBRATIONS = [
    "research_output/perception_validation/blue_hsv_v2.json",
    "research_output/perception_validation/robot1_yellow_hsv_v3.json",
]

bounds = []
for path in CALIBRATIONS:
    d = json.loads(Path(path).read_text())
    bounds.append((tuple(d["hsv_lower"]), tuple(d["hsv_upper"])))

segmenter = MultiHSVRangeSegmenter(
    hsv_ranges=tuple(bounds),
    min_component_pixels=80,
    keep_largest_components=3,
)

checkpoint = str(Path.home() / "sam2/checkpoints/sam2.1_hiera_tiny.pt")
config = "configs/sam2.1/sam2.1_hiera_t.yaml"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

predictor = SAM2ImagePredictor(
    build_sam2(config, checkpoint, device="cuda")
)

results = []

with h5py.File(HDF5, "r") as f:
    obs = f["data/demo_0/obs"]

    for frame in FRAMES:
        for camera in CAMERAS:
            rgb = np.asarray(obs[f"{camera}_rgb"][frame], dtype=np.uint8)
            depth = np.asarray(obs[f"{camera}_depth"][frame])[..., 0]
            valid = np.asarray(obs[f"{camera}_depth_valid"][frame])[..., 0]

            hsv_mask = segmenter.segment(rgb, depth, valid, camera)
            p = prompt(hsv_mask, pad=8)

            if p is None:
                sam_mask = np.zeros(rgb.shape[:2], dtype=bool)
            else:
                box, positive = p
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    predictor.set_image(rgb)
                    masks, _, _ = predictor.predict(
                        box=box,
                        point_coords=positive,
                        point_labels=np.array([1], dtype=np.int32),
                        multimask_output=False,
                    )
                sam_mask = np.asarray(masks[0], dtype=bool)

            state = estimate_garment_state(
                mask=sam_mask,
                depth_m=depth,
                depth_valid=valid,
                camera=camera,
            )

            row = {
                "frame": frame,
                "camera": camera,
                **state.to_dict(),
            }
            results.append(row)

            print(
                f"{frame} | {camera} | "
                f"valid={state.observation_valid} | "
                f"centroid={state.centroid_uv} | "
                f"depth={state.depth_median_m} | "
                f"depth_valid={state.depth_valid_ratio:.3f} | "
                f"fragmented={state.fragmentation_warning}"
            )

with (OUTPUT / "states.json").open("w") as fp:
    json.dump(results, fp, indent=2, allow_nan=False)

with (OUTPUT / "states.csv").open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", OUTPUT / "states.json")
print("Saved:", OUTPUT / "states.csv")
