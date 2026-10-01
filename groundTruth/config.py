from dataclasses import dataclass
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
SOURCES_YAML = MODULE_DIR / "config" / "groundtruth_sources.yaml"
DATA_DIR = MODULE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
NORMALIZED_DIR = DATA_DIR / "normalized"
RUNS_DIR = DATA_DIR / "runs"
REPORTS_DIR = DATA_DIR / "reports"
GROUND_TRUTH_CSV = DATA_DIR / "ground_truth.csv"

OBJECT_TYPES = ("curb_ramp", "stop_sign", "sign_other", "sidewalk")

# Attributes recorded per object_type; also the ground_truth.csv columns (blank when the source has no value).
CANONICAL_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "curb_ramp": ("running_slope_pct", "cross_slope_pct", "ramp_width_in", "gutter_lip_height_in"),
    "stop_sign": ("sign_code", "sign_size_in", "facing_direction", "side_of_street"),
    "sign_other": ("sign_code", "sign_size_in", "facing_direction", "side_of_street"),
    "sidewalk": ("width_in", "cross_slope_pct", "condition"),
}


@dataclass(frozen=True)
class Settings:
    sample_per_type: int = 150
    negative_per_type_fraction: float = 0.2  # negatives sampled as this fraction of the positive count
    pano_search_radius_m: float = 25
    negative_min_distance_m: float = 40
    negative_max_distance_m: float = 80
    negative_exclusion_radius_m: float = 30  # a candidate negative must have no same-type object within this
    sign_facing_tolerance_deg: float = 60
    date_gap_warn_years: float = 3
    train_fraction: float = 0.8
    random_seed: int = 0
    http_timeout: float = 30
    http_max_retries: int = 3
    request_delay_s: float = 0.2
