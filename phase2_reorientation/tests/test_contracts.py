"""Dependency-light tests for shared reorientation data contracts."""

import unittest

import numpy as np

from phase2_reorientation.contracts import (
    ContractError,
    GraspEvidence,
    ReorientationHandoff,
    ReorientationStatus,
    RuntimeObservation,
    neutral_action,
    validate_action,
)


def make_valid_observation():
    return {
        "rgb": np.zeros((5, 256, 256, 3), dtype=np.uint8),
        "depth": np.zeros((5, 256, 256, 1), dtype=np.float32),
        "depth_valid": np.zeros((5, 256, 256, 1), dtype=np.bool_),
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
        "simulation_time_s": np.asarray(0.05, dtype=np.float64),
    }


class ContractTests(unittest.TestCase):
    def test_valid_runtime_observation(self):
        observation = RuntimeObservation.from_mapping(make_valid_observation())
        self.assertEqual(observation.rgb.shape, (5, 256, 256, 3))
        self.assertEqual(observation.joint_position.shape, (2, 13))
        self.assertAlmostEqual(observation.simulation_time_s, 0.05)

    def test_missing_observation_field_is_rejected(self):
        observation = make_valid_observation()
        del observation["base_twist_world"]
        with self.assertRaises(ContractError):
            RuntimeObservation.from_mapping(observation)

    def test_wrong_shape_is_rejected(self):
        observation = make_valid_observation()
        observation["joint_position"] = np.zeros((26,), dtype=np.float32)
        with self.assertRaises(ContractError):
            RuntimeObservation.from_mapping(observation)

    def test_neutral_action_is_valid_float32(self):
        action = neutral_action()
        self.assertEqual(action.shape, (2, 9))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.all(validate_action(action) == 0.0))

    def test_invalid_action_is_rejected(self):
        action = neutral_action()
        action[0, 0] = 1.01
        with self.assertRaises(ContractError):
            validate_action(action)

    def test_handoff_requires_visual_grasp_evidence(self):
        common = dict(
            status=ReorientationStatus.SUCCESS,
            front_direction_world=np.array([1.0, 0.0, 0.0]),
            target_front_direction_world=np.array([1.0, 0.0, 0.0]),
            orientation_error_rad=0.0,
            orientation_confidence=0.95,
            gripper_positions_world=np.zeros((2, 3), dtype=np.float32),
            stable_duration_s=0.5,
            simulation_time_s=12.0,
        )
        unknown = ReorientationHandoff(
            grasp_evidence=GraspEvidence.UNKNOWN, **common
        )
        supported = ReorientationHandoff(
            grasp_evidence=GraspEvidence.BOTH_ARMS_VISUALLY_SUPPORTED, **common
        )
        readiness_limits = dict(
            min_orientation_confidence=0.60,
            max_orientation_error_rad=np.deg2rad(15.0),
            min_stable_duration_s=0.5,
        )
        self.assertFalse(unknown.ready_for_dressing(**readiness_limits))
        self.assertTrue(supported.ready_for_dressing(**readiness_limits))


if __name__ == "__main__":
    unittest.main()
