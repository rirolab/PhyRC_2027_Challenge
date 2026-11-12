"""Episode-local causal decision memory. Does not alter trusted priors online."""
from __future__ import annotations
import json
from pathlib import Path

class DecisionMemory:
    def __init__(self, out_dir, max_entries=1200):
        self.out=Path(out_dir);self.out.mkdir(parents=True,exist_ok=True)
        self.path=self.out/'strategy_decisions.jsonl'
        self.entries=[];self.max_entries=int(max_entries)
        self.failed=set()

    def record(self, event):
        if not isinstance(event,dict) or 'frame' not in event:raise ValueError('event needs frame')
        if self.entries and event['frame']<self.entries[-1]['frame']:
            raise ValueError('causal events must have monotonic frame')
        data=dict(event)
        self.entries.append(data)
        if len(self.entries)>self.max_entries:self.entries.pop(0)
        def native(v):
            if hasattr(v,'tolist'):return v.tolist()
            if hasattr(v,'item'):return v.item()
            raise TypeError(type(v))
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(data,default=native,allow_nan=False,ensure_ascii=False)+'\n')

    def failure(self,frame,scene_version,candidate_key,reason):
        key=(str(scene_version),str(candidate_key))
        self.failed.add(key)
        self.record(dict(frame=int(frame),event='CANDIDATE_REJECTED',scene_version=str(scene_version),
                         candidate_key=str(candidate_key),reason=str(reason)))

    def rejected_for_scene(self,scene_version):
        return {key for scene,key in self.failed if scene==str(scene_version)}

    def was_rejected(self,scene_version,candidate_key):
        return (str(scene_version),str(candidate_key)) in self.failed
