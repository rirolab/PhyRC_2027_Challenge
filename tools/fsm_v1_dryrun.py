#!/usr/bin/env python3
"""Read-only PhyRC HDF5 + sparse garment-state JSON -> FSM evidence trace.

This script is NOT a policy and never sends motion commands. A valid image mask
and gripper CLOSE request are not sufficient evidence to verify a grasp.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import h5py

# Works when executed from arbitrary working directory after extracting to repo.
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from phase2_reorientation.fsm_v1 import Evidence, ReorientationFSM, Phase


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hdf5',required=True,type=Path)
    ap.add_argument('--states',required=True,type=Path,help='states.json from frozen SAM + state estimation v1')
    ap.add_argument('--demo',default='demo_0')
    ap.add_argument('--output',type=Path,default=Path('research_output/fsm_v1_dryrun'))
    args=ap.parse_args()
    states=json.loads(args.states.read_text(encoding='utf8'))
    lookup={}
    for r in states:
        if r.get('camera') not in ('robot_0_head','robot_1_head'):continue
        key=(int(r['frame']),r['camera'])
        if key in lookup:raise ValueError(f'duplicate state {key}')
        lookup[key]=r
    frames=sorted({frame for frame,cam in lookup})
    fsm=ReorientationFSM()
    trace=[]
    with h5py.File(args.hdf5,'r') as f:
        obs=f[f'data/{args.demo}/obs']
        n=len(obs['simulation_time_s'])
        for frame in range(n):
            t=float(obs['simulation_time_s'][frame,0])
            a=lookup.get((frame,'robot_0_head'))
            b=lookup.get((frame,'robot_1_head'))
            both_states_valid=bool(a and b and a.get('observation_valid') is True and b.get('observation_valid') is True)
            cmd=obs['gripper_close_command'][frame]
            closed=bool(cmd[0] and cmd[1])
            # Only a *partial* camera observation from sparsely computed masks.
            e=Evidence(time_s=t, valid_cameras=both_states_valid,
                       both_grippers_commanded_closed=closed)
            phase=fsm.step(e)
            if frame in frames:
                trace.append({'frame':frame,'time_s':t,'state':phase.value,
                              'two_mask_states_valid':both_states_valid,
                              'grippers_commanded_closed':closed,
                              'visual_grasp_verified':False,
                              'note':'UNKNOWN grasp, plan and target: cannot progress without independent modules'})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'fsm_trace.json').write_text(json.dumps(trace,indent=2),encoding='utf8')
    with (args.output/'fsm_transitions.csv').open('w',newline='',encoding='utf8') as fp:
        writer=csv.DictWriter(fp,fieldnames=['at_s','before','after','reason']);writer.writeheader()
        for tr in fsm.transitions:writer.writerow(tr.__dict__)
    print('Dry-run finished. FINAL:',fsm.phase.value)
    for x in trace:print(x['frame'],x['state'],'both_masks_valid=',x['two_mask_states_valid'],
                          'close_intent=',x['grippers_commanded_closed'])
    print('Transitions:',len(fsm.transitions),'saved to',args.output)
    print('IMPORTANT: Unknown visual grasp/plan/target prevents success; no motion sent.')

if __name__=='__main__':main()
