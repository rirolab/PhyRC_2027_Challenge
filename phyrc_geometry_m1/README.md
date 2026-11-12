# PhyRC 2027 — Reorientation geometry M1

**Scope:** observation-only measurement + contract; **no actions and no automated trajectory**.

Inputs: PhyRC public `policy.hdf5` (`data/demo_0/obs`), robot geometry in official documented order. Base `xyz+wxyz` is world frame; gripper `xyz+wxyz` is base frame. Fingertip positions in base frame.

Test:
```bash
python -m unittest discover -s tests -p 'test_geometry_m1.py' -v
python tools/geometry_m1.py --hdf5 research_data/perception_validation/machine_01/policy.hdf5 --frames 1700 1730 1760 --output research_output/planning_geometry_m1
cat research_output/planning_geometry_m1/summary.csv
```

Optionally use `--base-proxy-radius VALUE` **only after validating the footprint radius in meters against official robot geometry**. This computes two *planar circles* clearance and **is not** collision detection for arms, wrists, cloth, manikin, or swept trajectories.

**Gate to M2:** obtain actual FK/collision geometry from permitted robot assets, align robot shape to base pose & joints, confirm explicit start/goal and plan collision constraints. Do not use `audit.hdf5` or privileged environment ground truth as policy input.

**Handoff:** reorientation preserves close intent; dressing module may change gripper intent after receiving documented `DRESSING_READY` contract. Geometry proof alone is insufficient for a verified goal.
