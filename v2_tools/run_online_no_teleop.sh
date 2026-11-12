#!/usr/bin/env bash
set -euo pipefail
REPO="${1:?Usage: bash run_online_no_teleop.sh /path/to/PhyRC_2027_Challenge [SEED] [SECONDS]}"
SEED="${2:-42}"
SECONDS_SIM="${3:-180}"
cd "$REPO"
python3 - <<'PY_CHECK'
import json
from pathlib import Path
p=Path('participant/v2_settings.json')
cfg=json.loads(p.read_text())
assert cfg['execution_mode']=='plan_only', 'REQUIRES PLAN_ONLY'
assert not cfg.get('sim_experiment_ack'), 'NO SIM MOTION ACK ALLOWED'
print('[CHECK] plan_only; manual scene side =',cfg.get('diagnostic_scene_side'))
PY_CHECK
mkdir -p output/v2_sdl
./run.sh python /project/v2_tools/v2_python_bootstrap.py /scripts/evaluate_policy.py \
 --policy /project/participant/policy.py \
 --checkpoint /project/participant/checkpoint.json \
 --seeds "$SEED" --seconds "$SECONDS_SIM" \
 --output /output/evaluation/v2_plan_only_smoke
# Runs only after official evaluator completes normally.
python3 v2_tools/inspect_plan_only.py --logs output/v2_sdl
python3 v2_tools/render_video.py --logs output/v2_sdl --fps 5
