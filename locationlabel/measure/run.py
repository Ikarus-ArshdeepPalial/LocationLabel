import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from common.progress import track_addresses

from .config import load_settings
from .depth import HFDepthModel
from .multiview import VggtReconstructor
from .stage import MeasurementStage

logger = logging.getLogger(__name__)

SUMMARY_COLUMNS = {
    "stop_sign_height_m": ("stop_sign_height", "value_m"),
    "stop_sign_height_source": ("stop_sign_height", "source"),
    "stop_sign_height_confidence": ("stop_sign_height", "confidence"),
    "curb_height_m": ("curb_height", "value_m"),
    "curb_height_source": ("curb_height", "source"),
    "curb_height_confidence": ("curb_height", "confidence"),
    "ramp_running_slope_pct": ("ramp", "running_slope_pct"),
    "ramp_cross_slope_pct": ("ramp", "cross_slope_pct"),
    "ramp_source": ("ramp", "source"),
    "ramp_confidence": ("ramp", "confidence"),
}


def run(addresses: list[str], force: bool = False, export_pointcloud: bool = False, summary_name: str | None = None) -> Counter[str]:
    from label_objs.models import select_device

    settings = load_settings()
    logger.info("Camera height %.2f m (±%.2f), stop sign width %.3f m", settings.camera_height_m, settings.camera_height_sigma_m, settings.stop_sign_width_m)
    device = select_device()
    stage = MeasurementStage(settings, HFDepthModel(settings.depth_model, device), force, export_pointcloud, VggtReconstructor(settings.vggt_model, device))
    sections = {address: stage.run(address) for address in track_addresses(addresses, "[4/4] measure")}
    if summary_name:
        update_summary(settings.output_dir / f"summary_{summary_name}.csv", sections)
    return Counter(section["status"] for section in sections.values())


def update_summary(path: Path, sections: dict[str, dict[str, Any]]) -> None:
    """Add measurement columns to the batch summary CSV (created by the geocode stage if it ran)."""
    rows: dict[str, dict[str, Any]] = {}
    fieldnames = ["input_address"]
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or fieldnames)
            rows = {row["input_address"]: row for row in reader}

    for address, section in sections.items():
        row = rows.setdefault(address, {"input_address": address})
        for column, (block, key) in SUMMARY_COLUMNS.items():
            row[column] = (section.get(block) or {}).get(key)
    fieldnames += [c for c in SUMMARY_COLUMNS if c not in fieldnames]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows.values())
    logger.info("Measurements added to %s", path)
