#!/usr/bin/env python3
"""Run inside PhyRC container via ./run.sh python /project/participant/../tools/preflight.py.
Prints blocking vs nonblocking checks without claiming Isaac trial success.
"""
import importlib
import json
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
repo=here.parent
participant=repo/'participant'
if str(participant) not in sys.path:sys.path.insert(0,str(participant))
checks={}
for name in ('numpy','scipy','embreex','trimesh','PIL'):
    try:importlib.import_module(name);checks[name]='OK'
    except Exception as exc:checks[name]=f'MISSING: {type(exc).__name__}: {exc}'
try:
    import mapper
    checks['camera_contract']=str(mapper.CONTRACT_PATH) if mapper.CONTRACT_PATH.exists() else 'MISSING'
    checks['camera_count']=len(mapper.CONFIG.get('cameras',[]))
    checks['v2_exchange']=str(mapper.PROJECT) if mapper._GEOMETRY_IMPORT_ERROR is None else f'MISSING: {mapper._GEOMETRY_IMPORT_ERROR}'
    if mapper._GEOMETRY_IMPORT_ERROR is None and mapper.CONTRACT_PATH.exists():
        g=mapper.Geometry()
        checks['robot_self_geometry']='OK'
except Exception as e:
    checks['robot_self_geometry']=f'FAILED: {repr(e)}'
try:
    import policy
    checks['entrypoint']='OK' if callable(policy.load_policy) else 'MISSING'
except Exception as e:
    checks['entrypoint']=f'FAILED: {repr(e)}'
settings=participant/'v2_settings.json'
checks['settings']=json.loads(settings.read_text()) if settings.exists() else 'MISSING'
if isinstance(checks.get('settings'),dict):
    cfg=checks['settings']
    checks['plan_only_lock']='OK' if cfg.get('execution_mode')=='plan_only' and not cfg.get('sim_experiment_ack') else 'FAILED_NOT_PLAN_ONLY'
    checks['diagnostic_scene_hint']='NOT_INFERRED' if cfg.get('diagnostic_scene_side') is not None else 'UNKNOWN_AUTOMATIC_SIDE'
    checks['video_capture']='OK' if cfg.get('video_enabled',False) else 'DISABLED'
print(json.dumps(checks,indent=2,ensure_ascii=False))
keys=('numpy','scipy','embreex','trimesh','PIL','entrypoint','robot_self_geometry','plan_only_lock','video_capture')
if any(checks.get(k)!='OK' for k in keys):
    print('PREFLIGHT FAIL: fix dependencies/geometry before online evaluation')
    sys.exit(2)
print('PREFLIGHT PASS: imports and robot geometry available (NOT a motion-validation PASS)')
