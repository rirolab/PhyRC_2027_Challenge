"""CPU verification: exported HDF5 vs recorded observations, commands and cameras."""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
ROOT=Path('/project') if Path('/project/config').exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/PhyRC_Sim'))
from Policy.teleop_recording import ArchiveReader
from Policy.dataset_reader import PolicyDataset
from Policy.training_dataset import training_observation

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('recording',type=Path)
p.add_argument('dataset',type=Path)
a=p.parse_args()
reader=ArchiveReader(a.recording)
public=PolicyDataset(a.dataset/'policy.hdf5')
count=0
with h5py.File(a.dataset/'policy.hdf5') as f, h5py.File(a.dataset/'audit.hdf5') as audit:
    assert f.attrs['complete'] and audit.attrs['complete']
    for header,values in reader.frames():
        if header['kind']!='policy_observation':
            continue
        demo=f['data/demo_'+str(int(values['policy_episode']))]
        review=audit[demo.name.split('/')[-1]]
        frame=int(values['policy_frame'])
        n=int(demo.attrs['num_samples'])
        for group,index,prefix in [('obs',frame,''),('next_obs',frame-1,'next_')]:
            if index<0 or index>=n:
                continue
            for key,value in values.items():
                if key.startswith('policy_obs/'):
                    np.testing.assert_array_equal(demo[group][key[len('policy_obs/'):]][index],value.reshape(-1))
            np.testing.assert_array_equal(review[prefix+'camera_world_from_optical'][index],
                                          values['policy_camera_world']@np.diag([1,-1,-1,1]))
            np.testing.assert_array_equal(review[prefix+'viewport_world'][index],values['camera_world'])
            np.testing.assert_array_equal(review[prefix+'viewport_lens'][index],values['camera_lens'])
            if prefix:
                np.testing.assert_array_equal(demo['actions'][index],values['policy_obs/previous_action'].reshape(18))
                np.testing.assert_array_equal(review['controller_targets_applied'][index],values['policy_applied_targets'])
                assert review['end_tick'][index]-review['start_tick'][index]==12
                count+=1
            for camera in public.contract['cameras']:
                rgb=demo[group][camera['id']+'_rgb'][index]
                depth=demo[group][camera['id']+'_depth'][index]
                mask=demo[group][camera['id']+'_depth_valid'][index]
                assert rgb.dtype==np.uint8 and np.ptp(rgb)>0
                assert depth.dtype==np.float32 and np.isfinite(depth).all()
                assert mask.dtype==bool and mask.any()
                assert np.all(depth[~mask]==0)
        if frame and frame<n:
            for key in demo['obs']:
                np.testing.assert_array_equal(demo['obs'][key][frame],demo['next_obs'][key][frame-1])
    assert count==len(public)
public.close()
print(json.dumps({'passed':True,'transitions':count,'exact_public_state_and_action_match':True,
                  'recorded_camera_match':True,'five_rgbd_cameras_valid':True,'obs_next_obs_continuity':True}))
