import unittest, numpy as np, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from geometry_m1 import rotate_wxyz,transform_point,check_snapshot,interpret_proxy

class TestGeometry(unittest.TestCase):
 def test_identity_rotation(self):
  self.assertTrue(np.allclose(rotate_wxyz([1,0,0,0],[1,2,3]),[1,2,3]))
 def test_quarter_turn(self):
  q=np.array([np.cos(np.pi/4),0,0,np.sin(np.pi/4)])
  self.assertTrue(np.allclose(rotate_wxyz(q,[1,0,0]),[0,1,0],atol=1e-7))
 def test_world_transform(self):
  self.assertTrue(np.allclose(transform_point([2,3,4,1,0,0,0],[.5,.5,.5]),[2.5,3.5,4.5]))
 def test_snapshot(self):
  bases=np.array([[0,0,0,1,0,0,0],[2,0,0,1,0,0,0]])
  grasps=np.array([[1,0,0,1,0,0,0],[1,0,0,1,0,0,0]])
  fingers=np.array([[[0,0,0],[.05,0,0]],[[0,0,0],[.05,0,0]]])
  s=check_snapshot(bases,grasps,fingers,[.05,.05],[True,True])
  self.assertTrue(s['geometry_valid']); self.assertAlmostEqual(s['base_xy_separation_m'],2)
  self.assertAlmostEqual(s['grasp_origin_separation_world_m'],2)
 def test_bad_quat(self):
  with self.assertRaises(ValueError): rotate_wxyz([0,0,0,0],[1,2,3])
 def test_proxy(self):
  self.assertTrue(interpret_proxy(.5,.25,.25,.1)['base_proxy_warning'])
  self.assertFalse(interpret_proxy(2,.25,.25,.1)['base_proxy_warning'])
if __name__=='__main__':unittest.main()
