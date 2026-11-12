# PhyRC V2 research baseline
Snapshot of the VM checkout based on lacie-life commit 049fb9f. Official upstream is rirolab/PhyRC_2027_Challenge. Research branch: dev/v2-baseline.

| Path | Responsibility |
| --- | --- |
| participant/policy.py, policyv2.py | Evaluator entry point and V1/V2 handoff |
| participant/original_v1.py, checkpoint.json | V1 pickup controller and tuning |
| participant/mapper.py, traversal_memory.py, traversability_evidence.py | RGB-D, footprint history and traversal fusion diagnostics |
| participant/strategy_decision.py, strategy_core.py, coupled_rrt_star.py | Strategy, coupled grid search and sampling planner |
| participant/v2_exchange | Geometry, calibration, forward kinematics and collision support. The USD link uses the identical upstream asset. |
| v2_tools | Bootstrap, evaluator launch, diagnostics and video utilities |
| phase2_reorientation*, phyrc_geometry_m1, tests, tools | Earlier perception, FSM, geometry experiments and checks |
| Root policy and servo scripts, scripts/v2* | Previous controllers and simulator probes |
| research_output/sam21_tiny_preview_20261112/run_preview.py | Reproducible SAM preview script; raw inputs and generated outputs stay on the VM |

The bootstrap requires the VM's installed V2 Python libraries. The tracked participant/.deps symlink resolves to /output/v2_sdl_pydeps inside the container. Restore the dependency directory under the VM output mount when moving to another VM; its installed contents are excluded from Git.

From a prepared Linux VM at the repository root:

```bash
bash v2_tools/run_online_no_teleop.sh "$PWD" 42 90
```

See participant/v2_settings.json for plan_only. Traversal fusion is diagnostic. Run outputs, recordings, caches, installed dependencies and historical backup copies are stored outside this baseline. For the optional DOF export in the simulator set STRETCH4_EXPORT_DOF_MAP=1. Keep main for upstream synchronization and branch from dev/v2-baseline for research.
