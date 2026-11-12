# FSM v1 — internal reorientation coordination, offline only

The finite-state machine is a high-level **supervisor**, not trajectory planning,
robot control, physical grasp detection or semantic cloth orientation.

Pipeline: frozen HSV+SAM -> state estimation -> visual grasp evidence +
orientation estimator -> safe trajectory planner -> controller -> feedback -> FSM.

### Contract

- `Evidence` is passed from independently validated modules.
- `valid_cameras` = two garment state observations considered valid; does NOT prove two grasps.
- `grasp_visually_supported`: `None` unknown, `False` disproven, `True` independently verified.
- `orientation_known`, `plan_safe`, `execution_started`, `execution_completed`,
  `target_verified` need external implementations. The script never fabricates these.
- `both_grippers_commanded_closed` is **diagnostic ONLY**. A close request is NOT a grasp.
- `MONITOR` responds to disappearing grasp evidence or safety clearance.
- `RECOVER` requires an externally verified restored grasp. It never invents a regrasp action.
- `SUCCESS` requires a sustained, independent verification of target orientation.
- `EXPLICIT HAZARD` immediately triggers RECOVER, not a controller motion instruction.
- Timeout and stability durations are internal pilot defaults, **not official scores**.

Do not combine 2D garment centroids from different camera frames as world xyz.
This v1 deliberately has no actuator connection and can NEVER swap positions
on its own; a world-frame path/IK/feedback controller is required.

### Install in repository

Extract ZIP at `~/PhyRC_2027_Challenge` root so it creates:
`phase2_reorientation/fsm_v1.py`, `tools/fsm_v1_dryrun.py`,
`tests/test_fsm_v1.py` (top-level `tests/` is okay).

### Tests

```bash
cd ~/PhyRC_2027_Challenge
python -m unittest discover -s tests -p 'test_fsm_v1.py' -v
```

### Real-data dry run (no GPU)

```bash
python tools/fsm_v1_dryrun.py \
  --hdf5 research_data/perception_validation/machine_01/policy.hdf5 \
  --states research_output/state_estimation_v1/states.json \
  --output research_output/fsm_v1_dryrun
```

Outputs `fsm_trace.json` and `fsm_transitions.csv`.
Given only 10 precomputed state observations and no independent grasp evidence,
FSM **will not** declare successful reorientation. That is intended and safe.

### What must be built before driving the robots

1. Validated evidence of two actual grasps using permitted observations.
2. Semantic orientation + target frame estimate (2D PCA is insufficient).
3. Collision, reachability, cloth tension-aware two-arm coordination planner.
4. Normalized `(2,9)` action controller with explicit feedback and safety supervisor.
5. Task success verification / hysteresis + recovery policy.

This repository's "Phase 2 reorientation" is an **internal** stage name; official
PhyRC Phase 2 refers to later physical robot evaluation.
