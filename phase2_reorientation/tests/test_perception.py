"""Unit tests for RGB-D perception primitives (NumPy only)."""

import unittest

import numpy as np

from phase2_reorientation.contracts import ContractError, RuntimeObservation
from phase2_reorientation.perception import (
    CAMERA_DEPTH_RANGES_M,
    CameraFrame,
    HSVRangeSegmenter,
    PerceptionProcessor,
)


class FixedMaskSegmenter:
    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=np.bool_)

    def segment(self, rgb, depth_m, depth_valid, camera_name):
        return self.mask.copy()


def make_valid_observation():
    return {
        "rgb": np.zeros((5, 256, 256, 3), dtype=np.uint8),
        "depth": np.ones((5, 256, 256, 1), dtype=np.float32),
        "depth_valid": np.ones((5, 256, 256, 1), dtype=np.bool_),
        "joint_position": np.zeros((2, 13), dtype=np.float32),
        "joint_velocity": np.zeros((2, 13), dtype=np.float32),
        "base_pose_world": np.zeros((2, 7), dtype=np.float32),
        "base_twist_world": np.zeros((2, 6), dtype=np.float32),
        "grasp_pose_base": np.zeros((2, 7), dtype=np.float32),
        "fingertip_position_base": np.zeros((2, 2, 3), dtype=np.float32),
        "fingertip_origin_distance": np.zeros((2, 1), dtype=np.float32),
        "controller_target": np.zeros((2, 9), dtype=np.float32),
        "gripper_close_command": np.zeros((2, 1), dtype=np.bool_),
        "previous_action": np.zeros((2, 9), dtype=np.float32),
        "simulation_time_s": np.asarray(1.25, dtype=np.float64),
    }


class PerceptionTests(unittest.TestCase):
    def test_no_segmenter_fails_closed(self):
        result = PerceptionProcessor().process(make_valid_observation())
        self.assertEqual(result.simulation_time_s, 1.25)
        self.assertEqual(len(result.detections), 5)
        self.assertEqual(result.valid_detections, ())
        self.assertEqual(
            result.get("overview").reason, "no_segmentation_provider_configured"
        )

    def test_pca_recovers_horizontal_image_axis_and_depth(self):
        mask = np.zeros((256, 256), dtype=np.bool_)
        mask[110:130, 60:190] = True
        observation = make_valid_observation()
        observation["depth"][:] = 2.0
        observation["depth"][0, 110:130, 60:190, 0] = 1.7
        processor = PerceptionProcessor(FixedMaskSegmenter(mask))
        result = processor.process(observation)
        detection = result.get("overview")

        self.assertTrue(detection.valid)
        self.assertEqual(detection.area_px, 20 * 130)
        self.assertEqual(detection.bbox_xyxy, (60, 110, 190, 130))
        self.assertAlmostEqual(float(detection.centroid_xy[0]), 124.5, places=3)
        self.assertAlmostEqual(float(detection.centroid_xy[1]), 119.5, places=3)
        self.assertAlmostEqual(abs(float(detection.principal_axis_xy[0])), 1.0, places=3)
        self.assertAlmostEqual(float(detection.median_depth_m), 1.7, places=5)
        self.assertIsNotNone(detection.angle_image_rad)

    def test_mask_too_small_is_invalid(self):
        mask = np.zeros((256, 256), dtype=np.bool_)
        mask[10:13, 10:13] = True
        detection = PerceptionProcessor(
            FixedMaskSegmenter(mask), min_mask_pixels=30
        ).process_frame(
            CameraFrame(
                "overview",
                np.zeros((256, 256, 3), dtype=np.uint8),
                np.ones((256, 256), dtype=np.float32),
                np.ones((256, 256), dtype=np.bool_),
            )
        )
        self.assertFalse(detection.valid)
        self.assertTrue(detection.reason.startswith("mask_too_small"))

    def test_invalid_depth_pixels_are_excluded_from_median(self):
        mask = np.zeros((256, 256), dtype=np.bool_)
        mask[80:120, 80:120] = True
        depth = np.zeros((256, 256), dtype=np.float32)
        valid = np.zeros((256, 256), dtype=np.bool_)
        depth[80:100, 80:120] = 1.0
        depth[100:120, 80:120] = 9.0  # invalid and must not affect median
        valid[80:100, 80:120] = True
        detection = PerceptionProcessor(FixedMaskSegmenter(mask)).process_frame(
            CameraFrame(
                "overview",
                np.zeros((256, 256, 3), dtype=np.uint8),
                depth,
                valid,
            )
        )
        self.assertTrue(detection.valid)
        self.assertAlmostEqual(detection.median_depth_m, 1.0, places=6)
        self.assertAlmostEqual(detection.depth_valid_fraction, 0.5, places=6)

    def test_bad_mask_shape_is_rejected(self):
        bad_mask = np.zeros((32, 32), dtype=np.bool_)
        processor = PerceptionProcessor(FixedMaskSegmenter(bad_mask))
        with self.assertRaises(ContractError):
            processor.process(make_valid_observation())

    def test_hsv_segmenter_selects_configured_red_pixels(self):
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        rgb[20:40, 30:50] = (255, 0, 0)
        depth = np.ones((256, 256), dtype=np.float32)
        valid = np.ones((256, 256), dtype=np.bool_)
        segmenter = HSVRangeSegmenter(
            hsv_lower=(350.0, 0.8, 0.8),
            hsv_upper=(10.0, 1.0, 1.0),
        )
        mask = segmenter.segment(rgb, depth, valid, "overview")
        self.assertEqual(int(mask.sum()), 400)

    def test_official_camera_depth_ranges_are_declared(self):
        self.assertEqual(CAMERA_DEPTH_RANGES_M["overview"], (0.05, 10.0))
        self.assertEqual(CAMERA_DEPTH_RANGES_M["robot_0_wrist"], (0.03, 3.0))
        self.assertEqual(len(CAMERA_DEPTH_RANGES_M), 5)

    def test_runtime_observation_contract_is_used(self):
        observation = RuntimeObservation.from_mapping(make_valid_observation())
        result = PerceptionProcessor().process(observation)
        self.assertEqual(result.simulation_time_s, observation.simulation_time_s)


if __name__ == "__main__":
    unittest.main()
