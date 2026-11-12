"""Shared data contracts for the PhyRC reorientation research pipeline.

This module deliberately contains data validation and shared types only.
It does not inspect simulator internals, infer a grasp from gripper intent,
or implement perception/planning/control algorithms.

Runtime observation shapes follow config/policy_interface.json schema 0.3.0.
The runtime RGB-D arrays are camera-stacked; HDF5 uses separate camera keys
and flattened non-image arrays, so HDF5 records must be adapted before being
passed to :class:`RuntimeObservation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


CONTRACTS_VERSION = "0.1.0"
NUM_ROBOTS = 2
JOINTS_PER_ROBOT = 13
NUM_CAMERAS = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
ACTION_SHAPE = (NUM_ROBOTS, 9)
ACTION_MIN = -1.0
ACTION_MAX = 1.0

CAMERA_ORDER: Tuple[str, ...] = (
    "overview",
    "robot_0_wrist",
    "robot_1_wrist",
    "robot_0_head",
    "robot_1_head",
)


class ContractError(ValueError):
    """Raised when data does not satisfy a shared pipeline contract."""


class ActionIndex(IntEnum):
    """Column indices for one robot's normalized action row."""

    BASE_FORWARD = 0
    BASE_LEFT = 1
    BASE_YAW = 2
    LIFT = 3
    ARM_EXTENSION = 4
    WRIST_YAW = 5
    WRIST_PITCH = 6
    WRIST_ROLL = 7
    GRIPPER = 8


class ReorientationPhase(str, Enum):
    """High-level phases of the reorientation state machine."""

    VERIFY_START = "verify_start"
    ESTIMATE_ORIENTATION = "estimate_orientation"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY_RESULT = "verify_result"
    SUCCESS = "success"
    ABORTED = "aborted"


class ReorientationStatus(str, Enum):
    """Result status, separate from the phase currently being executed."""

    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class GraspEvidence(str, Enum):
    """Strength of allowed evidence about the two-arm garment hold.

    `gripper_close_command` is only an intent signal. It is not proof of
    physical attachment. Therefore this enum intentionally has no status
    named `VERIFIED_BY_GRIPPER_COMMAND`.
    """

    UNKNOWN = "unknown"
    NOT_BOTH_SUPPORTED = "not_both_supported"
    BOTH_ARMS_VISUALLY_SUPPORTED = "both_arms_visually_supported"


@dataclass(frozen=True)
class RuntimeObservation:
    """Validated view of the public PhyRC runtime observation.

    Arrays retain the runtime shape and coordinate conventions from the
    official interface. Extra observation keys are ignored, not forwarded.
    """

    rgb: np.ndarray
    depth: np.ndarray
    depth_valid: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    base_pose_world: np.ndarray
    base_twist_world: np.ndarray
    grasp_pose_base: np.ndarray
    fingertip_position_base: np.ndarray
    fingertip_origin_distance: np.ndarray
    controller_target: np.ndarray
    gripper_close_command: np.ndarray
    previous_action: np.ndarray
    simulation_time_s: float

    @classmethod
    def from_mapping(
        cls,
        observation: Mapping[str, Any],
        *,
        copy_arrays: bool = False,
    ) -> "RuntimeObservation":
        """Validate and construct from the official runtime observation map.

        `copy_arrays=False` avoids copying large RGB-D tensors on every 20 Hz
        policy call. The returned instance should therefore be treated as
        read-only by downstream modules.
        """
        if not isinstance(observation, Mapping):
            raise ContractError(
                f"observation must be a mapping, got {type(observation).__name__}"
            )

        required = (
            "rgb",
            "depth",
            "depth_valid",
            "joint_position",
            "joint_velocity",
            "base_pose_world",
            "base_twist_world",
            "grasp_pose_base",
            "fingertip_position_base",
            "fingertip_origin_distance",
            "controller_target",
            "gripper_close_command",
            "previous_action",
            "simulation_time_s",
        )
        missing = [key for key in required if key not in observation]
        if missing:
            raise ContractError(f"observation is missing required fields: {missing}")

        arrays = {
            "rgb": _validated_array(
                observation["rgb"], "rgb", (5, 256, 256, 3), kind="uint"
            ),
            "depth": _validated_array(
                observation["depth"], "depth", (5, 256, 256, 1), kind="float"
            ),
            "depth_valid": _validated_array(
                observation["depth_valid"], "depth_valid", (5, 256, 256, 1), kind="bool"
            ),
            "joint_position": _validated_array(
                observation["joint_position"], "joint_position", (2, 13), kind="float"
            ),
            "joint_velocity": _validated_array(
                observation["joint_velocity"], "joint_velocity", (2, 13), kind="float"
            ),
            "base_pose_world": _validated_array(
                observation["base_pose_world"], "base_pose_world", (2, 7), kind="float"
            ),
            "base_twist_world": _validated_array(
                observation["base_twist_world"], "base_twist_world", (2, 6), kind="float"
            ),
            "grasp_pose_base": _validated_array(
                observation["grasp_pose_base"], "grasp_pose_base", (2, 7), kind="float"
            ),
            "fingertip_position_base": _validated_array(
                observation["fingertip_position_base"],
                "fingertip_position_base",
                (2, 2, 3),
                kind="float",
            ),
            "fingertip_origin_distance": _validated_array(
                observation["fingertip_origin_distance"],
                "fingertip_origin_distance",
                (2, 1),
                kind="float",
            ),
            "controller_target": _validated_array(
                observation["controller_target"], "controller_target", (2, 9), kind="float"
            ),
            "gripper_close_command": _validated_array(
                observation["gripper_close_command"],
                "gripper_close_command",
                (2, 1),
                kind="bool",
            ),
            "previous_action": _validated_array(
                observation["previous_action"], "previous_action", (2, 9), kind="float"
            ),
        }

        # RGB is specified as uint8. Accept unsigned integer input only when
        # all values are representable as uint8; normalize to uint8 here.
        rgb = arrays["rgb"]
        if rgb.dtype != np.uint8:
            if np.any(rgb > 255):
                raise ContractError("rgb contains values outside uint8 range [0, 255]")
            arrays["rgb"] = rgb.astype(np.uint8, copy=False)

        # Official depth values are optical-axis Z in metres; invalid pixels
        # use zero and are identified by depth_valid, not by value alone.
        if np.any(arrays["depth"] < 0.0):
            raise ContractError("depth cannot contain negative distances")

        sim_time = np.asarray(observation["simulation_time_s"])
        if sim_time.size != 1 or not np.issubdtype(sim_time.dtype, np.number):
            raise ContractError("simulation_time_s must be a numeric scalar")
        simulation_time_s = float(sim_time.reshape(()))
        if not np.isfinite(simulation_time_s) or simulation_time_s < 0.0:
            raise ContractError("simulation_time_s must be finite and non-negative")

        if copy_arrays:
            arrays = {key: value.copy() for key, value in arrays.items()}

        return cls(**arrays, simulation_time_s=simulation_time_s)


@dataclass(frozen=True)
class OrientationEstimate:
    """Estimated garment front direction in the world frame.

    Direction is a unit 3-vector in world coordinates (Z-up). A direction is
    absent when the estimate is invalid; confidence is always in [0, 1].
    """

    front_direction_world: Optional[np.ndarray]
    confidence: float
    simulation_time_s: float
    valid: bool
    source: str

    def __post_init__(self) -> None:
        confidence = float(self.confidence)
        time_s = float(self.simulation_time_s)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ContractError("orientation confidence must be in [0, 1]")
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ContractError("orientation timestamp must be finite and non-negative")

        if self.valid:
            if self.front_direction_world is None:
                raise ContractError("valid orientation estimate requires a direction")
            direction = _validated_array(
                self.front_direction_world, "front_direction_world", (3,), kind="float"
            ).astype(np.float64, copy=False)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                raise ContractError("front_direction_world cannot be a zero vector")
            object.__setattr__(self, "front_direction_world", direction / norm)
        elif self.front_direction_world is not None:
            direction = _validated_array(
                self.front_direction_world, "front_direction_world", (3,), kind="float"
            ).astype(np.float64, copy=False)
            object.__setattr__(self, "front_direction_world", direction)

        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "simulation_time_s", time_s)


@dataclass(frozen=True)
class SafetyReport:
    """Diagnostics produced by the safety layer; no simulator truth fields."""

    is_safe: bool
    reasons: Tuple[str, ...] = ()
    minimum_clearance_m: Optional[float] = None

    def __post_init__(self) -> None:
        if self.minimum_clearance_m is not None:
            clearance = float(self.minimum_clearance_m)
            if not np.isfinite(clearance) or clearance < 0.0:
                raise ContractError("minimum_clearance_m must be finite and non-negative")
            object.__setattr__(self, "minimum_clearance_m", clearance)
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))
        if self.is_safe and self.reasons:
            raise ContractError("a safe report cannot contain blocking reasons")


@dataclass(frozen=True)
class ReorientationHandoff:
    """Explicit output passed from reorientation to the later dressing stage.

    This is an internal project contract, not a PhyRC evaluator output.
    `BOTH_ARMS_VISUALLY_SUPPORTED` means supported by visual evidence only;
    it must not be described as simulator-confirmed physical attachment.
    """

    status: ReorientationStatus
    front_direction_world: Optional[np.ndarray]
    target_front_direction_world: Optional[np.ndarray]
    orientation_error_rad: Optional[float]
    orientation_confidence: float
    grasp_evidence: GraspEvidence
    gripper_positions_world: Optional[np.ndarray]
    stable_duration_s: float
    simulation_time_s: float
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        confidence = float(self.orientation_confidence)
        stable = float(self.stable_duration_s)
        sim_time = float(self.simulation_time_s)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ContractError("orientation_confidence must be in [0, 1]")
        if not np.isfinite(stable) or stable < 0.0:
            raise ContractError("stable_duration_s must be finite and non-negative")
        if not np.isfinite(sim_time) or sim_time < 0.0:
            raise ContractError("simulation_time_s must be finite and non-negative")

        for field_name in ("front_direction_world", "target_front_direction_world"):
            vector = getattr(self, field_name)
            if vector is not None:
                vector = _validated_array(vector, field_name, (3,), kind="float")
                norm = float(np.linalg.norm(vector))
                if norm < 1e-8:
                    raise ContractError(f"{field_name} cannot be a zero vector")
                object.__setattr__(self, field_name, vector / norm)

        if self.orientation_error_rad is not None:
            error = float(self.orientation_error_rad)
            if not np.isfinite(error) or error < 0.0 or error > np.pi:
                raise ContractError("orientation_error_rad must be in [0, pi]")
            object.__setattr__(self, "orientation_error_rad", error)

        if self.gripper_positions_world is not None:
            positions = _validated_array(
                self.gripper_positions_world, "gripper_positions_world", (2, 3), kind="float"
            )
            object.__setattr__(self, "gripper_positions_world", positions)

        object.__setattr__(self, "orientation_confidence", confidence)
        object.__setattr__(self, "stable_duration_s", stable)
        object.__setattr__(self, "simulation_time_s", sim_time)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    def ready_for_dressing(
        self,
        *,
        min_orientation_confidence: float,
        max_orientation_error_rad: float,
        min_stable_duration_s: float,
    ) -> bool:
        """Evaluate readiness using explicit project-configured thresholds.

        Thresholds are supplied by the caller (normally from `config.yaml`)
        rather than silently hard-coded in this shared contract.
        """
        confidence_threshold = float(min_orientation_confidence)
        error_threshold = float(max_orientation_error_rad)
        stable_threshold = float(min_stable_duration_s)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ContractError("min_orientation_confidence must be in [0, 1]")
        if not 0.0 <= error_threshold <= np.pi:
            raise ContractError("max_orientation_error_rad must be in [0, pi]")
        if stable_threshold < 0.0 or not np.isfinite(stable_threshold):
            raise ContractError("min_stable_duration_s must be finite and non-negative")

        return all(
            (
                self.status is ReorientationStatus.SUCCESS,
                self.front_direction_world is not None,
                self.target_front_direction_world is not None,
                self.orientation_error_rad is not None,
                self.orientation_error_rad <= error_threshold,
                self.orientation_confidence >= confidence_threshold,
                self.stable_duration_s >= stable_threshold,
                self.grasp_evidence is GraspEvidence.BOTH_ARMS_VISUALLY_SUPPORTED,
                self.gripper_positions_world is not None,
            )
        )


def validate_action(action: Any, *, copy_array: bool = False) -> np.ndarray:
    """Validate one runtime action and return float32 with shape (2, 9).

    This function rejects invalid actions instead of silently clipping or
    reinterpreting units. It does not scale normalized values into SI units.
    """
    array = np.asarray(action)
    if array.shape != ACTION_SHAPE:
        raise ContractError(f"action must have shape {ACTION_SHAPE}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ContractError(f"action must be numeric, got dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ContractError("action contains NaN or infinity")
    if np.any(array < ACTION_MIN) or np.any(array > ACTION_MAX):
        raise ContractError("action values must be within [-1, 1]")
    result = array.astype(np.float32, copy=copy_array or array.dtype != np.float32)
    return result


def neutral_action() -> np.ndarray:
    """Return a zero action. Gripper channel zero preserves prior intent."""
    return np.zeros(ACTION_SHAPE, dtype=np.float32)


def _validated_array(
    value: Any,
    field_name: str,
    expected_shape: Sequence[int],
    *,
    kind: str,
) -> np.ndarray:
    array = np.asarray(value)
    if tuple(array.shape) != tuple(expected_shape):
        raise ContractError(
            f"{field_name} must have shape {tuple(expected_shape)}, got {array.shape}"
        )

    if kind == "float":
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
            raise ContractError(f"{field_name} must have a numeric dtype, got {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise ContractError(f"{field_name} contains NaN or infinity")
        return array.astype(np.float32, copy=False)

    if kind == "uint":
        if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.signedinteger):
            raise ContractError(f"{field_name} must have an unsigned integer dtype")
        if np.any(array > 255):
            raise ContractError(f"{field_name} values must be within [0, 255]")
        return array

    if kind == "bool":
        if array.dtype != np.bool_:
            raise ContractError(f"{field_name} must have bool dtype, got {array.dtype}")
        return array

    raise RuntimeError(f"Unsupported validation kind: {kind}")
