import logging
from dataclasses import asdict
from datetime import datetime, timezone
from math import atan, degrees
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.progress import progress
from common.record import load_record, record_path, update_record
from cood_to_sview.geo import distance

from .calibration import Calibrator
from .config import IMAGE_EXTENSIONS, RAMP_FOLDER, Settings
from .crop import Crop, detect_ui_bars, margin_crop
from .exif import load_oriented, parse_exif
from .metadata import load_json, resolve_camera

logger = logging.getLogger(__name__)


def rotate_upright(image: np.ndarray, roll_deg: float) -> np.ndarray:
    """Rotate counter-clockwise by roll about the image centre (the assumed principal point)."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), roll_deg, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


class ImageImporter:
    """Stands in for Stage 2: copies the user's images and writes the same "streetview" record structure."""

    def __init__(self, settings: Settings, calibrators: list[Calibrator]) -> None:
        self._s = settings
        self._calibrators = calibrators

    def run(self, address: str, images_dir: Path) -> str:
        path = record_path(self._s.output_dir, address)
        try:
            section = self._import(load_record(path), images_dir, path.parent)
        except Exception as exc:
            logger.exception("Importing images for %r failed", address)
            section = {"status": "error", "source": "user_supplied", "message": f"Unexpected error: {exc}"}

        update_record(path, {"input_address": address, "streetview": section})
        log = logger.info if section["status"] == "ok" else logger.warning
        log("%r: user images %s - %s", address, section["status"], section.get("message", ""))
        return section["status"]

    def _import(self, record: dict[str, Any], images_dir: Path, address_dir: Path) -> dict[str, Any]:
        def listing(folder: Path) -> list[Path]:
            return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if folder.is_dir() else []

        # Photos in a `ramp` subfolder are extra views of one ramp, measured together; the rest are ordinary images.
        main_files, ramp_files = listing(images_dir), listing(images_dir / RAMP_FOLDER)
        files = [(f, "main") for f in main_files] + [(f, "ramp") for f in ramp_files]
        if not files:
            return {"status": "error", "source": "user_supplied", "message": f"No images found in {images_dir}"}

        target = {"lat": record["lat"], "lng": record["lng"]} if record.get("status") == "ok" else None
        folder = load_json(images_dir / "camera.json")
        (address_dir / "images").mkdir(parents=True, exist_ok=True)
        images, failed = [], []
        for file, group in progress(files, "  importing"):
            try:
                images.append(self._import_image(file, folder, address_dir / "images", target, group))
            except Exception as exc:
                logger.exception("Could not import %s", file)
                failed.append({"file": str(file), "error": str(exc)})

        located = [m for m in images if m["pano_lat"] is not None]
        return {
            "status": "ok" if images and not failed else "partial" if images else "error",
            "source": "user_supplied",
            "message": f"{len(images)}/{len(files)} images imported ({len(ramp_files)} ramp views), {len(located)} with a known camera position",
            "images_dir": str(images_dir),
            "source_type": self._s.source_type,
            "target": {"original": target, "adjusted": target, "method": "stage1"} if target else None,
            "panos": [{"pano_id": m["pano_id"], "lat": m["pano_lat"], "lng": m["pano_lng"], "role": "user"} for m in located],
            "images": images,
            "failed_images": failed,
            "image_requests": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _import_image(self, file: Path, folder: dict[str, Any], out_dir: Path, target: dict[str, float] | None, group: str = "main") -> dict[str, Any]:
        image, raw_exif = load_oriented(file)
        original_height, original_width = image.shape[:2]
        exif = parse_exif(raw_exif, original_width, original_height)
        camera = resolve_camera(image, load_json(file.with_suffix(".json")), folder, exif, self._s, self._calibrators)

        roll = camera.roll_deg.value
        straighten = group == "main" and abs(roll) >= self._s.min_roll_correction_deg  # ramp views keep their pixels untouched
        if straighten:
            image = rotate_upright(image, roll)

        crop = Crop()
        if self._s.crop_ui:
            crop = detect_ui_bars(image).union(margin_crop(self._s.defaults.ui_margins, original_width, original_height))
        image = crop.apply(image)
        height, width = image.shape[:2]

        filename = f"{RAMP_FOLDER}_{file.stem}.jpg" if group == "ramp" else f"{file.stem}.jpg"
        cv2.imwrite(str(out_dir / filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

        focal = camera.focal_px.value
        lat, lng = camera.position.value or (None, None)
        sources = camera.sources
        sources["principal_point"] = "image_center_crop_adjusted" if crop.as_tuple() != (0, 0, 0, 0) else "image_center"
        return {
            "filename": filename,
            "view": "user",
            "group": group,
            "role": "user",
            "pano_id": f"user:{file.stem}",
            "pano_lat": lat,
            "pano_lng": lng,
            "capture_date": exif.capture_date,
            "heading": round(camera.heading_deg.value, 2),
            "pitch": round(camera.pitch_deg.value, 2),
            "fov": round(degrees(2 * atan(width / 2 / focal)), 2),
            "width": width,
            "height": height,
            # The principal point stays at the original image centre, so cropping shifts it.
            "intrinsics": {"fx": focal, "fy": focal, "cx": original_width / 2 - crop.left, "cy": original_height / 2 - crop.top},
            "camera_height_m": camera.camera_height_m.value,
            "distance_to_target_m": round(distance(lat, lng, target["lat"], target["lng"]), 2) if lat is not None and target else None,
            "metadata_sources": sources,
            "calibrator": camera.calibrator,
            "roll_corrected_deg": round(roll, 2) if straighten else 0.0,
            "crop": asdict(crop),
            "original_file": str(file),
            "original_size": [original_width, original_height],
            "exif_orientation": exif.orientation,
        }
