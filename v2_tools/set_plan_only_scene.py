#!/usr/bin/env python3
"""Explicitly configure DIAGNOSTIC plan-only hint or no hint; cannot enable motion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--side',choices=['LEFT','RIGHT','UNKNOWN'],required=True)
a=p.parse_args()
path=a.repo/'participant'/'v2_settings.json'
if not path.exists():p.error(f'Missing {path}; install package first')
config=json.loads(path.read_text())
config['execution_mode']='plan_only';config['sim_experiment_ack']=False
config['diagnostic_scene_side']=a.side if a.side!='UNKNOWN' else None
config['diagnostic_ack']=a.side!='UNKNOWN'
config['video_enabled']=True
path.write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
print('Updated',path)
print('MODE: PLAN ONLY. Diagnostic side:',a.side)
print('IMPORTANT: manual side is NOT inferred from head/wrist sensors and cannot be used as validated online perception.')
