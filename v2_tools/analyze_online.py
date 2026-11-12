#!/usr/bin/env python3
"""Small-memory streamed summary of online V2 decision logs."""
import argparse
import collections
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path,help='Path to output/v2_sdl')
    args=parser.parse_args()
    files=sorted(args.output.glob('*/strategy_decisions.jsonl'))
    if not files:
        print('No decision logs found under',args.output)
        raise SystemExit(2)
    for log in files:
        counts=collections.Counter();modes=collections.Counter();errors=[];grasp=[]
        for line in log.open(encoding='utf-8'):
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            counts[event.get('event','UNKNOWN')]+=1
            if event.get('event')=='SDL_DECISION':modes[event.get('status','UNKNOWN')]+=1
            if event.get('event')=='V1_V2_HANDOFF':grasp.append(event.get('frame'))
            if event.get('event','').endswith('ERROR'):
                errors.append(dict(frame=event.get('frame'),error=event.get('reason')))
        print(json.dumps(dict(run=log.parent.name,events=dict(counts),
                    planner_outcomes=dict(modes),v1_handoff_frames=grasp,
                    errors=errors[:12]),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
