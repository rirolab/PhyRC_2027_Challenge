#!/usr/bin/env python3
"""Dependency-light smoke check under PhyRC's existing container Python."""
from pathlib import Path
import sys
sys.path.insert(0,'/project/participant')
import numpy as np
from traversal_memory import TraversalMemory
from traversability_evidence import fuse_traversal_with_rgbd

m=TraversalMemory((-4.5,-3.5),.05,(160,140))
for f in range(3):
    m.observe(f,f*.05,[[ -.35+.05*f,0.,0.,1.,0.,0.,0.],
                        [ .20, .05*f,0.,1.,0.,0.,0.]])
assert m.export()['base_pose_history'].shape==(3,2,7)
assert m.centerline_count_by_robot.sum()>0
shape=(160,140)
s=dict(base_cost_2d=np.full(shape,255,np.uint8))
r=dict(base_review_cost_2d=np.full(shape,250,np.uint8),
       base_review_obstacle_2d=np.zeros(shape,bool))
g=dict(grounded_cost_2d=np.full(shape,255,np.uint8))
a=fuse_traversal_with_rgbd(s,r,g,centerline_count_by_robot=m.centerline_count_by_robot,
                             footprint_count_by_robot=m.footprint_count_by_robot)
assert (s['base_cost_2d']==255).all(), 'STRICT map was modified'
assert a['traversal_fused_cost_2d'].min()==190
print('V4 SMOKE PASS; history',len(m.frames),'pose frames;',
      'centerline cells',np.count_nonzero(a['traversal_observed_centerline_2d']),
      '; STRICT unchanged; no actions sent')
