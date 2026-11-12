# Phase 2 Reorientation — Foundation

This directory contains the first two project foundation files:

- `config.yaml`: PhyRC interface constants plus explicitly labelled research defaults.
- `contracts.py`: validated runtime-observation, action, state, safety and dressing-handoff contracts.
- `tests/test_contracts.py`: lightweight checks for these contracts.

## Scope and assumptions

The runtime observation contract follows `config/policy_interface.json` schema `0.3.0`:

- Camera stack order: overview, robot 0 wrist, robot 1 wrist, robot 0 head, robot 1 head.
- Runtime RGB-D arrays are camera-stacked. HDF5 stores separate camera keys and flattened non-image arrays; convert those using the PhyRC dataset reader before using this runtime contract.
- World frame is right-handed, Z-up; pose quaternion order is `w, x, y, z`.
- Actions are normalized `float32` with shape `(2, 9)` and range `[-1, 1]`.
- Gripper command `0` means keep the previous open/close intent, not open.

`gripper_close_command` reports close intent, not physical grasp success. The handoff therefore requires a separate visual-evidence assessment and never treats the command bit as proof of attachment.

The orientation tolerance, confidence threshold, smoothing factor, stability duration and initial action caps in `config.yaml` are research starting values. They are not official PhyRC scoring thresholds. In particular, the competition's V-neck-facing-front score is evaluated in the dressing task after the head exits the neck opening; it is not a substitute for a target-direction estimator during this reorientation stage.

## Run the contract tests

From the parent directory containing `phase2_reorientation/`:

```bash
python3 -m unittest discover -s phase2_reorientation/tests -v
```

Requirements: Python 3.9+ and NumPy.

## Import examples

```python
from phase2_reorientation.contracts import (
    RuntimeObservation,
    neutral_action,
    validate_action,
)

runtime_obs = RuntimeObservation.from_mapping(obs)
action = validate_action(neutral_action())
```

The package does not yet load `config.yaml`; configuration loading will be added only when the first algorithm module needs it, so we do not introduce a YAML parser dependency prematurely.
