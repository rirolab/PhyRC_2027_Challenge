#!/usr/bin/env python3
"""Report whether V1 reached handoff and SDL chose a role/path. Never invent results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def analyze(run):
    log=run/'strategy_decisions.jsonl'
    if not log.exists():return dict(run=run.name,status='NO_POLICY_EVENTS')
    events=[]
    for line in log.read_text().splitlines():
        if line.strip():
            try:events.append(json.loads(line))
            except json.JSONDecodeError:pass
    handoff=[e for e in events if e.get('event')=='V1_V2_HANDOFF']
    intents=[e for e in events if e.get('event')=='SDL_ROLE_INTENT']
    plans=[e for e in events if e.get('event')=='SDL_DECISION']
    capture_errors=[e for e in events if e.get('event')=='VIDEO_CAPTURE_ERROR']
    errors=[e for e in events if e.get('event') in ('PLANNER_ERROR','MAPPING_ERROR')]
    role=intents[-1].get('intent',{}) if intents else (handoff[-1].get('conditional_intent',{}) if handoff else {})
    return dict(run=run.name,events=len(events),handoff=bool(handoff),
        handoff_frame=handoff[0].get('frame') if handoff else None,
        intent_status=role.get('status'),scene_side=role.get('garment_side'),
        scene_evidence=role.get('scene_evidence'),
        yielding_robot=role.get('yielding_robot'),passing_robot=role.get('passing_robot'),
        plan_attempts=len(plans),plans_with_base_path=sum(x.get('status')=='BASE_PATH_CANDIDATE' for x in plans),
        plan_status_latest=plans[-1].get('status') if plans else None,
        last_path_family=plans[-1].get('strategy') if plans else None,
        blocked_reasons=[e.get('reason') for e in events if e.get('event')=='PLAN_BLOCKED'][-5:],
        mapping_planning_errors=len(errors),video_errors=len(capture_errors),
        video_frames=len(list((run/'video_frames').glob('frame_*.jpg'))))


def main():
    a=argparse.ArgumentParser();a.add_argument('--logs',type=Path,default=Path('output/v2_sdl'))
    args=a.parse_args()
    rows=[analyze(p) for p in sorted(args.logs.iterdir()) if p.is_dir() and (p/'strategy_decisions.jsonl').exists()]
    if not rows:raise SystemExit(f'No strategy logs under {args.logs}')
    out=args.logs/'plan_only_report.json';out.write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    for row in rows:print(json.dumps(row,ensure_ascii=False))
    print('Saved:',out)

if __name__=='__main__':main()
