import os
from dataclasses import dataclass, field
from pathlib import Path

from common.settings import output_dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
RAMP_FOLDER = "ramp"  # photos of one ramp from different positions, in a subfolder of the images folder


@dataclass(frozen=True)
class SourceType:
    # Last-resort defaults, used only when no sidecar/EXIF/estimate provides the value.
    camera_height_m: float
    fov_deg: float
    # Crop applied with --crop-ui, as fractions of the image size (on top of detected uniform UI bars).
    ui_margins: dict[str, float] = field(default_factory=lambda: {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0})


SOURCE_TYPES = {
    # Street View car camera height (approximate); browser Street View's default view is roughly 90 degrees wide.
    "google_screenshot": SourceType(2.5, 90.0, {"top": 0.0, "bottom": 0.04, "left": 0.0, "right": 0.0}),
    # Handheld phone at chest height; a typical main camera (26 mm equivalent) is about 68 degrees wide.
    "phone": SourceType(1.5, 68.0),
}


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    source_type: str = "phone"
    camera_height_override: float | None = None
    crop_ui: bool = False
    # Roll below this is left alone; above it the image is rotated upright so later stages can assume no roll.
    min_roll_correction_deg: float = 0.5
    calibrators: tuple[str, ...] = ("geocalib", "lines")

    @property
    def defaults(self) -> SourceType:
        return SOURCE_TYPES[self.source_type]


def load_settings(source_type: str | None, camera_height: float | None, crop_ui: bool) -> Settings:
    return Settings(
        output_dir=output_dir(),
        source_type=source_type or os.getenv("IMAGE_SOURCE_TYPE", "phone"),
        camera_height_override=camera_height,
        crop_ui=crop_ui,
    )
