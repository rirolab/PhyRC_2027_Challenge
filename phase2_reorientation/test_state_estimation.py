"""Synthetic unit tests; no SAM GPU inference or HDF5 dependency."""
import numpy as np
from state_estimation import estimate_garment_state


def test_normal():
    m = np.zeros((256, 256), bool)
    m[30:50, 50:80] = True
    d = np.full((256, 256, 1), 1.20, np.float32)
    v = np.ones((256, 256, 1), bool)
    r = estimate_garment_state(m, d, v, 'robot_0_head')
    assert r.observation_valid
    assert r.centroid_uv == (64.5, 39.5)
    assert r.bbox_xyxy == (50, 30, 79, 49)
    assert abs(r.depth_median_m - 1.2) < 1e-6
    assert r.main_component_ratio == 1
    assert not r.fragmentation_warning


def test_fragmentation_advisory():
    m = np.zeros((256, 256), bool)
    m[10:30, 10:30] = True
    m[50:70, 50:70] = True
    d = np.ones((256, 256), np.float32)
    v = np.ones((256, 256), bool)
    r = estimate_garment_state(m, d, v, 'robot_0_head')
    assert r.fragmentation_warning and r.observation_valid
    assert r.main_component_ratio == 0.5


def test_missing_depth():
    m = np.zeros((256, 256), bool)
    m[10:30, 10:30] = True
    d = np.full((256, 256), np.nan)
    v = np.ones((256, 256), bool)
    r = estimate_garment_state(m, d, v, 'robot_1_head')
    assert not r.observation_valid and r.reason == 'insufficient_valid_depth'
    assert r.depth_median_m is None


def test_empty_mask():
    m = np.zeros((256, 256), bool)
    d = np.ones((256, 256), np.float32)
    v = np.ones((256, 256), bool)
    r = estimate_garment_state(m, d, v, 'robot_1_head')
    assert not r.observation_valid and r.reason == 'empty_mask'


if __name__ == '__main__':
    test_normal()
    test_fragmentation_advisory()
    test_missing_depth()
    test_empty_mask()
    print('PASS: 4 state-estimation synthetic tests')
