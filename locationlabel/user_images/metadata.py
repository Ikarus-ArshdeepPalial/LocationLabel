"""Resolve each camera parameter from the first available source: CLI, sidecar, camera.json, EXIF, estimate, default."""

import json
from dataclasses import dataclass
from math import radians, tan
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import Calibrator, calibrate
from .config import Settings
from .exif import ExifInfo


@dataclass
class Sourced:
    value: Any
    source: str


@dataclass
class CameraMetadata:
    focal_px: Sourced
    pitch_deg: Sourced
    roll_deg: Sourced
    heading_deg: Sourced
    camera_height_m: Sourced
    position: Sourced
    calibrator: str | None = None

    @property
    def sources(self) -> dict[str, str]:
        fields = ("focal_px", "pitch_deg", "roll_deg", "heading_deg", "camera_height_m", "position")
        return {name: getattr(self, name).source for name in fields}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def focal_from(values: dict[str, Any], width: int) -> float | None:
    if values.get("focal_px") is not None:
        return float(values["focal_px"])
    if values.get("fov_deg") is not None:
        return width / 2 / tan(radians(float(values["fov_deg"])) / 2)
    return None


def resolve_camera(
    image: np.ndarray,
    sidecar: dict[str, Any],
    folder: dict[str, Any],
    exif: ExifInfo,
    settings: Settings,
    calibrators: list[Calibrator],
) -> CameraMetadata:
    width = image.shape[1]
    layers = (("sidecar", sidecar), ("camera_json", folder))

    def from_layers(key: str) -> Sourced | None:
        return next((Sourced(float(values[key]), source) for source, values in layers if values.get(key) is not None), None)

    focal = next((Sourced(f, source) for source, values in layers if (f := focal_from(values, width))), None)
    if focal is None and exif.focal_px:
        focal = Sourced(exif.focal_px, "exif")
    pitch, roll = from_layers("pitch_deg"), from_layers("roll_deg")

    calibrator = None
    if focal is None or pitch is None or roll is None:
        found = calibrate(calibrators, image, focal.value if focal else None)
        if found:
            estimate, calibrator = found
            focal = focal or (Sourced(estimate.focal_px, "estimated") if estimate.focal_px else None)
            pitch = pitch or Sourced(estimate.pitch_deg, "estimated")
            roll = roll or Sourced(estimate.roll_deg, "estimated")

    defaults = settings.defaults
    heading = from_layers("heading_deg") or (Sourced(exif.heading_deg, "exif") if exif.heading_deg is not None else None)
    height = Sourced(settings.camera_height_override, "cli") if settings.camera_height_override is not None else from_layers("camera_height_m")
    position = next(
        (Sourced((float(v["lat"]), float(v["lng"])), source) for source, v in layers if v.get("lat") is not None and v.get("lng") is not None),
        Sourced((exif.lat, exif.lng), "exif") if exif.lat is not None else Sourced(None, "unknown"),
    )
    return CameraMetadata(
        focal_px=focal or Sourced(width / 2 / tan(radians(defaults.fov_deg) / 2), "default"),
        pitch_deg=pitch or Sourced(0.0, "default"),
        roll_deg=roll or Sourced(0.0, "default"),
        heading_deg=heading or Sourced(0.0, "default"),
        camera_height_m=height or Sourced(defaults.camera_height_m, "default"),
        position=position,
        calibrator=calibrator,
    )
