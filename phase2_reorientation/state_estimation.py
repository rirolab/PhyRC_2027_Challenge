"""Image-plane garment state from an already computed segmentation mask.

No segmentation, calibration, privileged state, pose or front/back inference here.
Depth statistics assume input is in metres and are NOT a 3-D world pose.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional
import numpy as np

DEPTH_RANGES_M = {
    'overview': (0.05, 10.0),
    'robot_0_wrist': (0.03, 3.0),
    'robot_1_wrist': (0.03, 3.0),
    'robot_0_head': (0.05, 5.0),
    'robot_1_head': (0.05, 5.0),
}


@dataclass(frozen=True)
class GarmentImageState:
    camera: str
    observation_valid: bool
    reason: str
    centroid_uv: Optional[tuple[float, float]]
    bbox_xyxy: Optional[tuple[int, int, int, int]]
    mask_pixels: int
    mask_area_ratio: float
    main_component_ratio: float
    depth_valid_ratio: float
    depth_median_m: Optional[float]
    depth_p10_m: Optional[float]
    depth_p90_m: Optional[float]
    fragmentation_warning: bool
    depth_warning: bool

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_garment_state(
    mask: np.ndarray,
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
    camera: str,
    *,
    min_mask_pixels: int = 80,
    min_depth_valid_ratio: float = 0.30,
    fragmentation_threshold: float = 0.85,
) -> GarmentImageState:
    """Estimate 2-D geometry and masked depth, retaining fragmented visible cloth.

    Strict validity: mask >= min_mask_pixels; depth valid ratio >= threshold.
    A fragmented mask is advisory, NOT a reason to discard the observation.
    ``depth_m`` is assumed to be in metres; confirm upstream contract first.
    """
    if camera not in DEPTH_RANGES_M:
        raise ValueError(f'Unknown camera: {camera!r}')
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f'mask must have shape (H,W), got {m.shape}')
    if m.dtype != np.bool_:
        if not np.issubdtype(m.dtype, np.number) or not np.all((m == 0) | (m == 1)):
            raise ValueError('mask must be boolean or 0/1 binary')
    m = m.astype(bool, copy=False)
    d = np.asarray(depth_m)
    v = np.asarray(depth_valid)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if v.ndim == 3 and v.shape[-1] == 1:
        v = v[..., 0]
    if d.shape != m.shape or v.shape != m.shape:
        raise ValueError(f'shape mismatch: mask={m.shape}, depth={d.shape}, valid={v.shape}')
    if not np.issubdtype(d.dtype, np.number):
        raise ValueError('depth_m must be numeric')
    if not (0 <= min_depth_valid_ratio <= 1) or not (0 <= fragmentation_threshold <= 1):
        raise ValueError('ratios must be within [0,1]')
    if min_mask_pixels < 1:
        raise ValueError('min_mask_pixels must be positive')

    y, x = np.nonzero(m)
    area = int(x.size)
    frac = float(area/m.size)
    if area == 0:
        return GarmentImageState(camera, False, 'empty_mask', None, None, 0, 0.0,
                                 0.0, 0.0, None, None, None, False, True)

    # No cv2 dependency: 8-neighbour connected-component measurement by scipy
    # is avoidable here; import OpenCV, which is already part of the pipeline.
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    main = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    main_frac = float(main / area)

    low, high = DEPTH_RANGES_M[camera]
    valid_depth = (m & v.astype(bool) & np.isfinite(d) &
                   (d >= low) & (d <= high))
    depth_samples = np.asarray(d[valid_depth], dtype=np.float64)
    ratio = float(depth_samples.size / area)
    depth_good = ratio >= min_depth_valid_ratio
    mask_good = area >= min_mask_pixels
    reason = ('ok' if mask_good and depth_good else
              'mask_too_small' if not mask_good else 'insufficient_valid_depth')

    return GarmentImageState(
        camera=camera, observation_valid=bool(mask_good and depth_good), reason=reason,
        centroid_uv=(float(x.mean()), float(y.mean())),
        bbox_xyxy=(int(x.min()), int(y.min()), int(x.max()), int(y.max())),
        mask_pixels=area, mask_area_ratio=frac, main_component_ratio=main_frac,
        depth_valid_ratio=ratio,
        depth_median_m=float(np.median(depth_samples)) if depth_samples.size else None,
        depth_p10_m=float(np.percentile(depth_samples, 10)) if depth_samples.size else None,
        depth_p90_m=float(np.percentile(depth_samples, 90)) if depth_samples.size else None,
        fragmentation_warning=bool(main_frac < fragmentation_threshold),
        depth_warning=not depth_good,
    )
