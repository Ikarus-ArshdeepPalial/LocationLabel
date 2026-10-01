import os
from dataclasses import dataclass, field
from pathlib import Path

from common.settings import env_path, output_dir


@dataclass(frozen=True)
class Limits:
    # LEGAL limit, 2010 ADA Standards 405.2: curb ramp running slope not steeper than 1:12.
    ramp_running_slope_pct: float = 8.33
    # LEGAL limit, 2010 ADA Standards 405.3: cross slope not steeper than 1:48.
    ramp_cross_slope_pct: float = 2.08
    # LEGAL minimum, MUTCD: 7 ft to the sign bottom where parking or pedestrians are expected (5 ft rural).
    stop_sign_min_height_m: float = 2.13
    # CHOSEN cap, not a legal limit (there is no federal maximum). Reported when a sign can't be measured.
    stop_sign_default_max_m: float = 3.05
    # CHOSEN cap, not a legal limit (there is no federal maximum). Reported when a curb can't be measured.
    curb_default_max_m: float = 0.229


@dataclass(frozen=True)
class Plausible:
    """Physically plausible ranges; estimates outside them are discarded as errors, never clipped."""

    stop_sign_height_m: tuple[float, float] = (1.0, 4.0)
    curb_height_m: tuple[float, float] = (0.0, 0.5)
    ramp_slope_pct: tuple[float, float] = (0.0, 30.0)


# Confidence multiplier for an estimate relying on a camera parameter from this source (user-supplied images).
METADATA_PENALTY = {"estimated": 0.85, "default": 0.7}

# How strongly each method depends on each camera parameter (exponent on the penalty). The reference-scale
# method barely depends on intrinsics: focal length cancels (distance ~ f, angle ~ 1/f), and pitch mostly does.
METHOD_DEPENDENCIES: dict[str, dict[str, float]] = {
    "flat_ground": {"camera_height_m": 1.0, "pitch_deg": 1.0, "focal_px": 1.0},
    "reference_scale": {"pitch_deg": 0.3, "focal_px": 0.1},
    "point_cloud": {"camera_height_m": 0.5, "pitch_deg": 1.0, "focal_px": 1.0},
    "triangulation": {"camera_height_m": 1.0, "pitch_deg": 1.0, "focal_px": 0.5, "heading_deg": 1.0, "position": 1.0},
}


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    # Street View car camera height above the road; approximate, so it carries its own uncertainty.
    camera_height_m: float = 2.5
    camera_height_sigma_m: float = 0.15
    pixel_sigma: float = 1.5
    # Stop sign face width: 30 in on residential streets, 36 in on multi-lane roads.
    stop_sign_width_m: float = 0.762
    max_sign_aspect_deviation: float = 0.35
    depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
    depth_ground_max_distance_m: float = 15.0
    depth_scale_warn: float = 0.25
    sidewalk_max_range_m: float = 12.0  # farther than this the depth is too coarse to measure a sidewalk across
    curb_neighborhood_m: float = 3.0
    max_curb_face_px: int = 60
    ransac_threshold_m: float = 0.03
    ransac_iterations: int = 300
    min_triangulation_angle_deg: float = 10.0
    group_radius_m: float = 4.0
    mad_k: float = 3.0
    confidence_threshold: float = 0.5
    # Spread at which the "agreement" confidence factor drops to 0.5, per measurement.
    agreement_tolerance: dict[str, float] = field(
        default_factory=lambda: {"stop_sign_height": 0.15, "curb_height": 0.03, "ramp_slope": 1.5}
    )
    # Ramp slope from several photos of one ramp (a `ramp` folder next to your images). VGGT's original checkpoint is
    # non-commercial; for a product use facebook/VGGT-1B-Commercial (gated).
    vggt_model: str = "facebook/VGGT-1B"
    multiview_max_views: int = 6
    multiview_crop_factor: float = 2.0
    # JSON list of regional limits (see measure/limits.py); no file means no limit checks.
    limits_file: Path | None = None
    near_distance_m: float = 5.0
    far_distance_m: float = 30.0
    limits: Limits = Limits()
    plausible: Plausible = Plausible()


def load_settings() -> Settings:
    return Settings(
        output_dir=output_dir(),
        camera_height_m=float(os.getenv("CAMERA_HEIGHT_M", "2.5")),
        stop_sign_width_m=float(os.getenv("STOP_SIGN_WIDTH_M", "0.762")),
        depth_model=os.getenv("DEPTH_MODEL", Settings.depth_model),
        vggt_model=os.getenv("VGGT_MODEL", Settings.vggt_model),
        multiview_max_views=int(os.getenv("MULTIVIEW_MAX_VIEWS", Settings.multiview_max_views)),
        multiview_crop_factor=float(os.getenv("MULTIVIEW_CROP_FACTOR", Settings.multiview_crop_factor)),
        limits_file=env_path("LIMITS_FILE", None),
        confidence_threshold=float(os.getenv("MEASURE_CONFIDENCE_THRESHOLD", "0.5")),
    )
