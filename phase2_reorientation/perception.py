"""RGB-D garment perception primitives for Phase 2 reorientation.

This module deliberately separates *garment segmentation* from geometric
measurement. The official policy observation provides five stacked RGB-D
cameras, but it does not provide ground-truth garment masks, cloth state,
object poses, or audit camera transforms. Consequently, this module never
reads those privileged sources and never silently pretends that a depth image
is a garment segmentation.

The initial implementation provides:
- validation and camera-wise extraction of the public RGB-D observations;
- a configurable HSV-range segmenter for controlled baseline experiments;
- image-plane centroid, bounding box, principal axis and masked depth stats;
- explicit invalid/low-confidence results when segmentation is unavailable.

The principal axis is an *undirected 2-D image-plane axis*. It is not the
shirt's semantic front direction and is not a world-frame vector. Resolving
front/back (e.g. detecting the V-neck) and transforming it into world
coordinates require separate, calibrated/learned stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

from .contracts import CAMERA_ORDER, ContractError, RuntimeObservation


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# Official camera clipping ranges documented in the PhyRC data guide.
CAMERA_DEPTH_RANGES_M: Mapping[str, Tuple[float, float]] = {
    "overview": (0.05, 10.0),
    "robot_0_wrist": (0.03, 3.0),
    "robot_1_wrist": (0.03, 3.0),
    "robot_0_head": (0.05, 5.0),
    "robot_1_head": (0.05, 5.0),
}


class SegmentationProvider(Protocol):
    """Interface for a garment mask provider.

    Implementations receive only one camera's allowed RGB-D observation and
    return a 2-D mask where True denotes pixels believed to belong to the
    garment. A learned segmentation model can implement this same interface.
    """

    def segment(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        depth_valid: np.ndarray,
        camera_name: str,
    ) -> np.ndarray:
        """Return a boolean garment mask of shape ``(H, W)``."""


@dataclass(frozen=True)
class HSVRangeSegmenter:
    """Simple configurable RGB-to-HSV threshold segmenter.

    This is a *controlled-baseline tool*, not a universal T-shirt detector.
    Set thresholds from representative images before relying on its output.
    Hue is in degrees [0, 360]; saturation/value are in [0, 1]. If
    ``hsv_lower[0] > hsv_upper[0]``, the hue interval wraps through 360°.
    Optional depth limits are in metres and are applied only where depth is
    valid. No threshold values are guessed by default; callers must supply
    them explicitly.
    """

    hsv_lower: Tuple[float, float, float]
    hsv_upper: Tuple[float, float, float]
    min_depth_m: Optional[float] = None
    max_depth_m: Optional[float] = None

    def __post_init__(self) -> None:
        lower = np.asarray(self.hsv_lower, dtype=np.float64)
        upper = np.asarray(self.hsv_upper, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("HSV bounds must each contain exactly 3 values")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("HSV bounds must be finite")
        if not (0.0 <= lower[0] <= 360.0 and 0.0 <= upper[0] <= 360.0):
            raise ValueError("Hue bounds must be in [0, 360]")
        if np.any(lower[1:] < 0.0) or np.any(lower[1:] > 1.0):
            raise ValueError("Lower saturation/value bounds must be in [0, 1]")
        if np.any(upper[1:] < 0.0) or np.any(upper[1:] > 1.0):
            raise ValueError("Upper saturation/value bounds must be in [0, 1]")
        if self.min_depth_m is not None and (
            not np.isfinite(self.min_depth_m) or self.min_depth_m < 0.0
        ):
            raise ValueError("min_depth_m must be finite and non-negative")
        if self.max_depth_m is not None and (
            not np.isfinite(self.max_depth_m) or self.max_depth_m <= 0.0
        ):
            raise ValueError("max_depth_m must be finite and positive")
        if (
            self.min_depth_m is not None
            and self.max_depth_m is not None
            and self.min_depth_m >= self.max_depth_m
        ):
            raise ValueError("min_depth_m must be less than max_depth_m")

    def segment(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        depth_valid: np.ndarray,
        camera_name: str,
    ) -> np.ndarray:
        del camera_name  # The same explicit HSV thresholds are used per call.
        image = _validate_rgb(rgb)
        depth, valid = _validate_depth_pair(depth_m, depth_valid, image.shape[:2])
        hsv = _rgb_to_hsv(image)
        lower = np.asarray(self.hsv_lower, dtype=np.float32)
        upper = np.asarray(self.hsv_upper, dtype=np.float32)

        if lower[0] <= upper[0]:
            hue_match = (hsv[..., 0] >= lower[0]) & (hsv[..., 0] <= upper[0])
        else:
            hue_match = (hsv[..., 0] >= lower[0]) | (hsv[..., 0] <= upper[0])

        mask = (
            hue_match
            & (hsv[..., 1] >= lower[1])
            & (hsv[..., 1] <= upper[1])
            & (hsv[..., 2] >= lower[2])
            & (hsv[..., 2] <= upper[2])
        )
        if self.min_depth_m is not None:
            mask &= valid & (depth >= self.min_depth_m)
        if self.max_depth_m is not None:
            mask &= valid & (depth <= self.max_depth_m)
        return mask.astype(np.bool_, copy=False)



@dataclass(frozen=True)
class MultiHSVRangeSegmenter:
    """Garment segmenter using the union of multiple HSV ranges.

    Typical Phase-2 use:
        blue fabric mask
        UNION
        yellow fabric mask

    After union, small disconnected components are removed. This is still a
    deterministic baseline; it does not infer shirt semantics or front/back
    orientation.

    ``hsv_ranges`` contains pairs:
        ((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))

    Hue is circular in [0, 360]. A range with h_lo > h_hi wraps through 360°.

    ``min_component_pixels`` removes isolated colour blobs.

    ``keep_largest_components`` can optionally keep only the N largest valid
    components. None means retain every component satisfying the minimum size.
    """

    hsv_ranges: Tuple[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]], ...
    ]

    min_component_pixels: int = 80
    keep_largest_components: Optional[int] = 3
    min_depth_m: Optional[float] = None
    max_depth_m: Optional[float] = None

    def __post_init__(self) -> None:
        if len(self.hsv_ranges) < 1:
            raise ValueError("hsv_ranges must contain at least one HSV range")

        for lower_tuple, upper_tuple in self.hsv_ranges:
            lower = np.asarray(lower_tuple, dtype=np.float64)
            upper = np.asarray(upper_tuple, dtype=np.float64)

            if lower.shape != (3,) or upper.shape != (3,):
                raise ValueError("each HSV bound must contain exactly 3 values")

            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                raise ValueError("HSV bounds must be finite")

            if not (0.0 <= lower[0] <= 360.0):
                raise ValueError("lower hue must be in [0, 360]")
            if not (0.0 <= upper[0] <= 360.0):
                raise ValueError("upper hue must be in [0, 360]")

            if np.any(lower[1:] < 0.0) or np.any(lower[1:] > 1.0):
                raise ValueError("lower saturation/value must be in [0, 1]")
            if np.any(upper[1:] < 0.0) or np.any(upper[1:] > 1.0):
                raise ValueError("upper saturation/value must be in [0, 1]")

        if self.min_component_pixels < 1:
            raise ValueError("min_component_pixels must be >= 1")

        if (
            self.keep_largest_components is not None
            and self.keep_largest_components < 1
        ):
            raise ValueError("keep_largest_components must be >= 1 or None")

        if self.min_depth_m is not None and (
            not np.isfinite(self.min_depth_m) or self.min_depth_m < 0.0
        ):
            raise ValueError("min_depth_m must be finite and non-negative")

        if self.max_depth_m is not None and (
            not np.isfinite(self.max_depth_m) or self.max_depth_m <= 0.0
        ):
            raise ValueError("max_depth_m must be finite and positive")

        if (
            self.min_depth_m is not None
            and self.max_depth_m is not None
            and self.min_depth_m >= self.max_depth_m
        ):
            raise ValueError("min_depth_m must be less than max_depth_m")

    @staticmethod
    def _range_mask(
        hsv: np.ndarray,
        lower_tuple: Tuple[float, float, float],
        upper_tuple: Tuple[float, float, float],
    ) -> np.ndarray:

        lower = np.asarray(lower_tuple, dtype=np.float32)
        upper = np.asarray(upper_tuple, dtype=np.float32)

        if lower[0] <= upper[0]:
            hue_match = (
                (hsv[..., 0] >= lower[0])
                & (hsv[..., 0] <= upper[0])
            )
        else:
            # Circular Hue interval, e.g. 340° -> 20°.
            hue_match = (
                (hsv[..., 0] >= lower[0])
                | (hsv[..., 0] <= upper[0])
            )

        return (
            hue_match
            & (hsv[..., 1] >= lower[1])
            & (hsv[..., 1] <= upper[1])
            & (hsv[..., 2] >= lower[2])
            & (hsv[..., 2] <= upper[2])
        )

    @staticmethod
    def _filter_components(
        mask: np.ndarray,
        min_component_pixels: int,
        keep_largest_components: Optional[int],
    ) -> np.ndarray:
        """8-connected component filter implemented without OpenCV/SciPy."""

        from collections import deque

        source = np.asarray(mask, dtype=np.bool_)
        height, width = source.shape

        visited = np.zeros_like(source, dtype=np.bool_)
        components = []

        neighbours = (
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        )

        for y0, x0 in zip(*np.nonzero(source)):
            if visited[y0, x0]:
                continue

            queue = deque([(int(y0), int(x0))])
            visited[y0, x0] = True
            pixels = []

            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))

                for dy, dx in neighbours:
                    ny = y + dy
                    nx = x + dx

                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and source[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(pixels) >= min_component_pixels:
                components.append(pixels)

        if not components:
            return np.zeros_like(source)

        components.sort(key=len, reverse=True)

        if keep_largest_components is not None:
            components = components[:keep_largest_components]

        filtered = np.zeros_like(source)

        for pixels in components:
            ys, xs = zip(*pixels)
            filtered[
                np.asarray(ys, dtype=np.intp),
                np.asarray(xs, dtype=np.intp),
            ] = True

        return filtered

    def segment(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        depth_valid: np.ndarray,
        camera_name: str,
    ) -> np.ndarray:
        del camera_name

        image = _validate_rgb(rgb)
        depth, valid = _validate_depth_pair(
            depth_m,
            depth_valid,
            image.shape[:2],
        )

        hsv = _rgb_to_hsv(image)

        # Union all configured garment colour ranges.
        mask = np.zeros(
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            dtype=np.bool_,
        )

        for lower, upper in self.hsv_ranges:
            mask |= self._range_mask(hsv, lower, upper)

        # Optional RGB-D gating.
        if self.min_depth_m is not None:
            mask &= valid & (depth >= self.min_depth_m)

        if self.max_depth_m is not None:
            mask &= valid & (depth <= self.max_depth_m)

        # Remove small isolated colour regions before geometry/PCA.
        mask = self._filter_components(
            mask,
            min_component_pixels=self.min_component_pixels,
            keep_largest_components=self.keep_largest_components,
        )

        return mask.astype(np.bool_, copy=False)


@dataclass(frozen=True)
class CameraFrame:
    """One camera's public RGB-D frame, with invalid depth masked explicitly."""

    camera_name: str
    rgb: np.ndarray
    depth_m: np.ndarray
    depth_valid: np.ndarray

    def __post_init__(self) -> None:
        if self.camera_name not in CAMERA_ORDER:
            raise ContractError(f"unknown camera name: {self.camera_name}")
        rgb = _validate_rgb(self.rgb)
        depth, valid = _validate_depth_pair(self.depth_m, self.depth_valid, rgb.shape[:2])
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "depth_valid", valid)


@dataclass(frozen=True)
class GarmentDetection:
    """Geometric measurement of a segmented garment in one camera image.

    ``bbox_xyxy`` uses pixel coordinates with an exclusive maximum boundary.
    ``principal_axis_xy`` is a unit vector in image coordinates (x right,
    y down), and is sign-canonicalized only for reproducible logging. Since
    the axis has no semantic polarity, it cannot distinguish shirt-front from
    shirt-back.
    """

    camera_name: str
    valid: bool
    mask: np.ndarray
    area_px: int
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    centroid_xy: Optional[np.ndarray]
    principal_axis_xy: Optional[np.ndarray]
    principal_axis_ratio: Optional[float]
    median_depth_m: Optional[float]
    depth_valid_fraction: float
    geometry_confidence: float
    reason: str = ""

    @property
    def angle_image_rad(self) -> Optional[float]:
        """Principal-axis angle in image coordinates, modulo pi."""
        if not self.valid or self.principal_axis_xy is None:
            return None
        x, y = self.principal_axis_xy
        return float(np.arctan2(y, x) % np.pi)


@dataclass(frozen=True)
class PerceptionResult:
    """Camera-wise garment perception output for one policy observation."""

    simulation_time_s: float
    detections: Mapping[str, GarmentDetection]

    @property
    def valid_detections(self) -> Tuple[GarmentDetection, ...]:
        return tuple(
            detection
            for name in CAMERA_ORDER
            if (detection := self.detections.get(name)) is not None and detection.valid
        )

    def get(self, camera_name: str) -> Optional[GarmentDetection]:
        return self.detections.get(camera_name)


class PerceptionProcessor:
    """Convert a public PhyRC observation into per-camera garment geometry.

    ``segmenter=None`` is intentionally fail-closed: it returns invalid
    detections with an explicit reason rather than treating every valid depth
    pixel or every image pixel as the garment.
    """

    def __init__(
        self,
        segmenter: Optional[SegmentationProvider] = None,
        *,
        min_mask_pixels: int = 30,
        min_depth_valid_fraction: float = 0.10,
    ) -> None:
        if min_mask_pixels < 3:
            raise ValueError("min_mask_pixels must be >= 3")
        if not 0.0 <= min_depth_valid_fraction <= 1.0:
            raise ValueError("min_depth_valid_fraction must be in [0, 1]")
        self.segmenter = segmenter
        self.min_mask_pixels = int(min_mask_pixels)
        self.min_depth_valid_fraction = float(min_depth_valid_fraction)

    def process(
        self,
        observation: Union[RuntimeObservation, Mapping[str, object]],
    ) -> PerceptionResult:
        """Run perception on one official runtime observation.

        The input mapping must use the runtime stacked arrays ``rgb``
        ``(5,256,256,3)``, ``depth`` and ``depth_valid``. This is not the
        flattened/separate-camera HDF5 layout.
        """
        runtime = (
            observation
            if isinstance(observation, RuntimeObservation)
            else RuntimeObservation.from_mapping(observation)
        )

        detections: Dict[str, GarmentDetection] = {}
        for index, camera_name in enumerate(CAMERA_ORDER):
            frame = CameraFrame(
                camera_name=camera_name,
                rgb=runtime.rgb[index],
                depth_m=runtime.depth[index, ..., 0],
                depth_valid=runtime.depth_valid[index, ..., 0],
            )
            detections[camera_name] = self._process_frame(frame)

        return PerceptionResult(
            simulation_time_s=runtime.simulation_time_s,
            detections=detections,
        )

    def process_frame(self, frame: CameraFrame) -> GarmentDetection:
        """Process one camera frame; useful for unit tests and offline tools."""
        if not isinstance(frame, CameraFrame):
            raise TypeError("frame must be a CameraFrame")
        return self._process_frame(frame)

    def _process_frame(self, frame: CameraFrame) -> GarmentDetection:
        if self.segmenter is None:
            return _invalid_detection(
                frame.camera_name,
                reason="no_segmentation_provider_configured",
            )

        raw_mask = self.segmenter.segment(
            frame.rgb, frame.depth_m, frame.depth_valid, frame.camera_name
        )
        mask_array = np.asarray(raw_mask)
        if mask_array.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ContractError(
                "segmentation mask for "
                f"{frame.camera_name} must have shape {(IMAGE_HEIGHT, IMAGE_WIDTH)}, "
                f"got {mask_array.shape}"
            )
        if mask_array.dtype != np.bool_:
            if not np.issubdtype(mask_array.dtype, np.number):
                raise ContractError("segmentation mask must be bool or numeric")
            if not np.all(np.isfinite(mask_array)):
                raise ContractError("segmentation mask contains NaN or infinity")
            mask_array = mask_array > 0.5
        mask = mask_array.astype(np.bool_, copy=False)

        ys, xs = np.nonzero(mask)
        area = int(xs.size)
        if area < self.min_mask_pixels:
            return _invalid_detection(
                frame.camera_name,
                mask=mask,
                area_px=area,
                reason=f"mask_too_small:{area}<{self.min_mask_pixels}",
            )

        x_min = int(xs.min())
        x_max = int(xs.max()) + 1
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1
        centroid = np.array([xs.mean(), ys.mean()], dtype=np.float64)

        # PCA of pixel coordinates gives the dominant *undirected* image axis.
        points = np.column_stack((xs, ys)).astype(np.float64, copy=False)
        centered = points - centroid
        covariance = (centered.T @ centered) / max(area - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        major = max(float(eigenvalues[order[0]]), 0.0)
        minor = max(float(eigenvalues[order[1]]), 0.0)
        axis = eigenvectors[:, order[0]].astype(np.float64, copy=True)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12 or major <= 1e-9:
            return _invalid_detection(
                frame.camera_name,
                mask=mask,
                area_px=area,
                bbox_xyxy=(x_min, y_min, x_max, y_max),
                centroid_xy=centroid,
                reason="degenerate_mask_geometry",
            )
        axis /= axis_norm
        # Canonical sign only; this does not assign front/back semantics.
        if axis[0] < 0.0 or (abs(axis[0]) <= 1e-12 and axis[1] < 0.0):
            axis *= -1.0

        valid_depth_pixels = mask & frame.depth_valid
        depth_count = int(np.count_nonzero(valid_depth_pixels))
        depth_fraction = float(depth_count / area)
        median_depth = (
            float(np.median(frame.depth_m[valid_depth_pixels]))
            if depth_count
            else None
        )

        ratio = float(minor / major) if major > 0.0 else 1.0
        area_fraction = float(area / (IMAGE_HEIGHT * IMAGE_WIDTH))
        area_score = min(1.0, area / float(self.min_mask_pixels * 4))
        # Very large masks are likely to include background/clutter.
        if area_fraction > 0.85:
            area_score *= max(0.0, (1.0 - area_fraction) / 0.15)
        anisotropy_score = float(np.clip(1.0 - ratio, 0.0, 1.0))
        depth_score = (
            0.5 + 0.5 * depth_fraction
            if depth_fraction >= self.min_depth_valid_fraction
            else 0.5 * depth_fraction / max(self.min_depth_valid_fraction, 1e-8)
        )
        geometry_confidence = float(
            np.clip(area_score * anisotropy_score * depth_score, 0.0, 1.0)
        )

        return GarmentDetection(
            camera_name=frame.camera_name,
            valid=True,
            mask=mask,
            area_px=area,
            bbox_xyxy=(x_min, y_min, x_max, y_max),
            centroid_xy=centroid,
            principal_axis_xy=axis,
            principal_axis_ratio=ratio,
            median_depth_m=median_depth,
            depth_valid_fraction=depth_fraction,
            geometry_confidence=geometry_confidence,
            reason="",
        )


def _invalid_detection(
    camera_name: str,
    *,
    mask: Optional[np.ndarray] = None,
    area_px: int = 0,
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    centroid_xy: Optional[np.ndarray] = None,
    reason: str,
) -> GarmentDetection:
    empty_mask = (
        np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.bool_)
        if mask is None
        else np.asarray(mask, dtype=np.bool_)
    )
    return GarmentDetection(
        camera_name=camera_name,
        valid=False,
        mask=empty_mask,
        area_px=int(area_px),
        bbox_xyxy=bbox_xyxy,
        centroid_xy=centroid_xy,
        principal_axis_xy=None,
        principal_axis_ratio=None,
        median_depth_m=None,
        depth_valid_fraction=0.0,
        geometry_confidence=0.0,
        reason=reason,
    )


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ContractError(
            f"RGB image must have shape {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}, got {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ContractError(f"RGB image must be uint8, got {image.dtype}")
    return image


def _validate_depth_pair(
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    expected_hw: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m)
    valid = np.asarray(depth_valid)
    expected = tuple(expected_hw)
    if depth.shape == expected + (1,):
        depth = depth[..., 0]
    if valid.shape == expected + (1,):
        valid = valid[..., 0]
    if depth.shape != expected:
        raise ContractError(f"depth must have shape {expected}, got {depth.shape}")
    if valid.shape != expected:
        raise ContractError(f"depth_valid must have shape {expected}, got {valid.shape}")
    if not np.issubdtype(depth.dtype, np.number) or np.issubdtype(depth.dtype, np.bool_):
        raise ContractError("depth must have a numeric dtype")
    if not np.all(np.isfinite(depth)):
        raise ContractError("depth contains NaN or infinity")
    if np.any(depth < 0.0):
        raise ContractError("depth cannot contain negative distances")
    if valid.dtype != np.bool_:
        raise ContractError("depth_valid must have bool dtype")
    return depth.astype(np.float32, copy=False), valid


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to float32 HSV (H degrees, S/V in [0,1])."""
    image = rgb.astype(np.float32) / 255.0
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    maximum = np.max(image, axis=-1)
    minimum = np.min(image, axis=-1)
    delta = maximum - minimum

    hue = np.zeros_like(maximum, dtype=np.float32)
    nonzero = delta > 1e-8
    red_max = nonzero & (maximum == r)
    green_max = nonzero & ~red_max & (maximum == g)
    blue_max = nonzero & ~red_max & ~green_max
    hue[red_max] = (60.0 * ((g[red_max] - b[red_max]) / delta[red_max])) % 360.0
    hue[green_max] = 60.0 * ((b[green_max] - r[green_max]) / delta[green_max] + 2.0)
    hue[blue_max] = 60.0 * ((r[blue_max] - g[blue_max]) / delta[blue_max] + 4.0)

    saturation = np.zeros_like(maximum, dtype=np.float32)
    nonblack = maximum > 1e-8
    saturation[nonblack] = delta[nonblack] / maximum[nonblack]
    return np.stack((hue, saturation, maximum), axis=-1)
