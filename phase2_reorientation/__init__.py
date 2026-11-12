"""Research package for PhyRC dual-robot garment reorientation."""

from .contracts import (
    ACTION_SHAPE,
    ActionIndex,
    ContractError,
    GraspEvidence,
    OrientationEstimate,
    ReorientationHandoff,
    ReorientationPhase,
    ReorientationStatus,
    RuntimeObservation,
    SafetyReport,
    neutral_action,
    validate_action,
)
from .perception import (
    CAMERA_DEPTH_RANGES_M,
    CameraFrame,
    GarmentDetection,
    HSVRangeSegmenter,
    PerceptionProcessor,
    PerceptionResult,
    SegmentationProvider,
)

__all__ = [
    "ACTION_SHAPE",
    "ActionIndex",
    "CAMERA_DEPTH_RANGES_M",
    "CameraFrame",
    "ContractError",
    "GarmentDetection",
    "GraspEvidence",
    "HSVRangeSegmenter",
    "OrientationEstimate",
    "PerceptionProcessor",
    "PerceptionResult",
    "ReorientationHandoff",
    "ReorientationPhase",
    "ReorientationStatus",
    "RuntimeObservation",
    "SafetyReport",
    "SegmentationProvider",
    "neutral_action",
    "validate_action",
]
