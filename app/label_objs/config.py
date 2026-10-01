import os
from dataclasses import dataclass, field
from pathlib import Path

from common.settings import CACHE_DIR, env_path, output_dir

WEIGHTS_DIR = CACHE_DIR / "weights"

# Canonical class -> segmentation model label names (Mapillary Vistas). stop_sign comes from the detector.
CLASS_MAP: dict[str, list[str]] = {
    "road": ["Road", "Service Lane", "Bike Lane", "Lane Marking - General"],
    "sidewalk": ["Sidewalk"],
    "curb": ["Curb"],
    "curb_ramp": ["Curb Cut"],
    "pole": ["Pole", "Traffic Sign Frame", "Utility Pole"],
    "traffic_sign": ["Traffic Sign (Front)", "Traffic Sign (Back)"],
    "crosswalk": ["Crosswalk - Plain", "Lane Marking - Crosswalk"],
    "fire_hydrant": ["Fire Hydrant"],
    "pedestrian_area": ["Pedestrian Area"],
    "vegetation": ["Vegetation"],
    "terrain": ["Terrain"],
    "building": ["Building"],
    "car": ["Car"],
}

# Index 0 is "unlabeled" in the semantic PNG; canonical classes follow in this order.
CLASSES: list[str] = ["unlabeled", *CLASS_MAP, "stop_sign"]

COLORS: dict[str, tuple[int, int, int]] = {  # RGB
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "curb": (255, 170, 0),
    "curb_ramp": (0, 255, 255),
    "pole": (255, 255, 0),
    "traffic_sign": (220, 220, 0),
    "crosswalk": (255, 255, 255),
    "fire_hydrant": (255, 0, 0),
    "pedestrian_area": (140, 140, 200),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "building": (70, 70, 70),
    "car": (0, 0, 142),
    "stop_sign": (255, 0, 64),
}

MIN_AREA: dict[str, int] = {
    "road": 2000,
    "sidewalk": 800,
    "curb": 150,
    "curb_ramp": 150,
    "pole": 60,
    "traffic_sign": 60,
    "crosswalk": 400,
    "fire_hydrant": 60,
    "pedestrian_area": 800,
    "vegetation": 2000,
    "terrain": 800,
    "building": 2000,
    "car": 800,
}

# Large "stuff" classes are shown by colour only; everything else also gets a text label.
UNLABELED_IN_ANNOTATION = {"road", "sidewalk", "vegetation", "terrain", "building", "pedestrian_area"}
CURB_LINE_CLASSES = {"curb", "curb_ramp"}
ROAD_SIDE_CLASSES = {"road", "crosswalk"}
TARGET_CLASSES = {"stop_sign", "curb_ramp"}

# What Stage 3 can detect, chosen with --detect (or DETECT in .env). Each target uses the model that tested best for it:
# ramps = RampNet + the segmenter's big regions, driveways = Grounding DINO + SAM, signs = YOLO, poles/curbs = the segmenter.
TARGETS = ("ramps", "driveways", "signs", "poles", "curbs")


def parse_targets(text: str) -> tuple[str, ...]:
    """'ramps,driveways' -> ('ramps', 'driveways'); 'all' or empty -> every target. Signs need poles to find the pole holding them."""
    names = [t.strip().lower() for t in text.split(",") if t.strip()]
    chosen = set(TARGETS) if not names or "all" in names else set(names)
    unknown = chosen - set(TARGETS)
    if unknown:
        raise ValueError(f"unknown detection target(s) {sorted(unknown)}; choose from {list(TARGETS)} or 'all'")
    if "signs" in chosen:
        chosen.add("poles")
    return tuple(t for t in TARGETS if t in chosen)


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    cache_path: Path
    seg_model: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    # Higher than the default because ramps are small in the frame; on RampNet panoramas 1024 found ~9 points more.
    seg_input_size: int = 1024
    detector_model: str = "yolo11x.pt"
    stop_sign_threshold: float = 0.4
    # Two sources of curb ramps, each strong where the other is weak.
    # 1) A segmented curb-cut region counts on its own if it is big and confident: this is what finds the driveway
    #    aprons in close-up photos. Thin low-confidence strips along a curb do not pass.
    ramp_min_area_fraction: float = 0.001
    ramp_min_confidence: float = 0.55
    # 2) A RampNet point counts at this score. It finds far and small pedestrian ramps (on 160 rendered views 0.30
    #    found 66% with 0.04 extras per ramp-free view) but scores at most 0.19 on close-up driveway photos, so this
    #    stays above that. A point also lets a weaker segmented region under it count.
    ramp_min_score: float = 0.30
    # A segmented ramp region is used to measure a ramp if it lies within this fraction of the image diagonal of it.
    ramp_region_radius: float = 0.06
    # Driveways: a text-prompted detector finds them and each one becomes a ramp, since every driveway meets the street at
    # an apron. On close-up photos it scored 0.85-0.88; on a wide view it scored 0.2-0.42 and merged the driveway with the
    # sidewalk; the floor is 0.2 by choice, so expect false driveways on wide scenes.
    driveway_prompt: str = "a driveway."
    driveway_min_score: float = 0.2
    driveway_min_area_fraction: float = 0.005
    grounding_model: str = "IDEA-Research/grounding-dino-tiny"
    sam_model: str = "facebook/sam-vit-base"
    batch_size: int = 4
    min_area: dict[str, int] = field(default_factory=lambda: dict(MIN_AREA))


def load_settings(seg_model: str | None = None) -> Settings:
    return Settings(
        output_dir=output_dir(),
        cache_path=env_path("DETECT_CACHE_PATH", CACHE_DIR / "detect.sqlite"),
        seg_model=seg_model or os.getenv("SEG_MODEL", Settings.seg_model),
        seg_input_size=int(os.getenv("SEG_INPUT_SIZE", Settings.seg_input_size)),
        detector_model=os.getenv("DETECTOR_MODEL", Settings.detector_model),
        stop_sign_threshold=float(os.getenv("STOP_SIGN_THRESHOLD", "0.4")),
        ramp_min_area_fraction=float(os.getenv("RAMP_MIN_AREA_FRACTION", Settings.ramp_min_area_fraction)),
        ramp_min_confidence=float(os.getenv("RAMP_MIN_CONFIDENCE", Settings.ramp_min_confidence)),
        ramp_min_score=float(os.getenv("RAMP_MIN_SCORE", Settings.ramp_min_score)),
        driveway_min_score=float(os.getenv("DRIVEWAY_MIN_SCORE", Settings.driveway_min_score)),
        driveway_min_area_fraction=float(os.getenv("DRIVEWAY_MIN_AREA_FRACTION", Settings.driveway_min_area_fraction)),
        batch_size=int(os.getenv("DETECT_BATCH_SIZE", "4")),
    )
