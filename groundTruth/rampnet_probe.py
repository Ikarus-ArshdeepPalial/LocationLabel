"""Can RampNet's panorama model find ramps in ordinary perspective photos, and how wrong can the camera guess be?

The view is placed at its (assumed) position in an otherwise empty equirectangular canvas. Scored like the Mask2Former
benchmark: a labeled ramp is found if a peak lands within HIT_RADIUS_PX of it; extras are peaks in ramp-free views.
"""

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .rampbench import BENCH_DIR, HIT_RADIUS_PX, VIEW_FOV, VIEW_PITCH, VIEW_SIZE, build_views, project

logger = logging.getLogger(__name__)

THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55)
PANO_HW = (2048, 4096)
FILL = (124, 116, 104)  # ImageNet mean colour, so empty canvas is neutral to the network
# name -> (pitch error in degrees, field-of-view factor) applied to what we tell the placement about the camera
VARIANTS = {"exact camera": (0.0, 1.0), "pitch 10 deg off": (10.0, 1.0), "field of view 25% too narrow": (0.0, 0.75)}
DISTANCE_BANDS = ((4, 7), (7, 10), (10, 14))


def view_to_panorama(view_rgb: np.ndarray, yaw_deg: float, pitch_deg: float, fov_deg: float) -> np.ndarray:
    """Places a perspective view where it belongs in an equirectangular canvas; everything outside it stays neutral."""
    h, w = PANO_HW
    lon = ((np.arange(w) + 0.5) / w - 0.5) * 2 * np.pi
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lon, lat = np.meshgrid(lon, lat)
    dx, dy, dz = np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)

    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    x, z2 = dx * math.cos(yaw) - dz * math.sin(yaw), dx * math.sin(yaw) + dz * math.cos(yaw)
    y, z = dy * math.cos(pitch) - z2 * math.sin(pitch), dy * math.sin(pitch) + z2 * math.cos(pitch)

    f = (VIEW_SIZE / 2) / math.tan(math.radians(fov_deg) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        map_x = np.where(z > 1e-6, VIEW_SIZE / 2 + f * x / z, -1).astype(np.float32)
        map_y = np.where(z > 1e-6, VIEW_SIZE / 2 - f * y / z, -1).astype(np.float32)
    return cv2.remap(view_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=FILL)


def peaks_in_view(peaks, yaw: float, pitch: float, fov: float) -> list[dict]:
    """Panorama-pixel peaks converted back to view pixels with the same assumed camera, dropping those outside it."""
    out = []
    for p in peaks:
        px = project(p.x / PANO_HW[1], p.y / PANO_HW[0], yaw, pitch, VIEW_SIZE, fov)
        if px:
            out.append({"x": px[0], "y": px[1], "score": p.score})
    return out


def detect_all(views: list) -> dict[str, dict[str, list[dict]]]:
    import torch

    from label_objs.rampnet import PanoRampDetector

    detector = PanoRampDetector("cuda:0" if torch.cuda.is_available() else "cpu")
    found: dict[str, dict[str, list[dict]]] = {name: {} for name in VARIANTS}
    for i, view in enumerate(views, 1):
        rgb = cv2.cvtColor(cv2.imread(str(BENCH_DIR / "views" / f"{view.name}.jpg")), cv2.COLOR_BGR2RGB)
        for name, (pitch_error, fov_factor) in VARIANTS.items():
            pitch, fov = VIEW_PITCH + pitch_error, VIEW_FOV * fov_factor
            found[name][view.name] = peaks_in_view(detector.detect(view_to_panorama(rgb, view.yaw, pitch, fov), THRESHOLDS[0]), view.yaw, pitch, fov)
        if i % 20 == 0:
            logger.info("Detected %d/%d views", i, len(views))
    return found


def hit(view, peaks: list[dict], threshold: float) -> bool:
    return any(p["score"] >= threshold and math.hypot(p["x"] - view.gt_pixel[0], p["y"] - view.gt_pixel[1]) <= HIT_RADIUS_PX for p in peaks)


def score(views: list, peaks_by_view: dict[str, list[dict]], threshold: float, positives=None) -> tuple[float, float]:
    positives = positives if positives is not None else [v for v in views if v.kind == "positive"]
    negatives = [v for v in views if v.kind == "negative"]
    found = sum(hit(v, peaks_by_view[v.name], threshold) for v in positives if v.gt_pixel)
    extras = sum(sum(p["score"] >= threshold for p in peaks_by_view[v.name]) for v in negatives)
    return found / max(len(positives), 1), extras / max(len(negatives), 1)


def run_probe(rescore: bool = False) -> Path:
    views = build_views(60, 0)
    cache = BENCH_DIR / "rampnet_peaks_v2.json"
    if cache.exists() and not rescore:
        found = json.loads(cache.read_text())
    else:
        found = detect_all(views)
        cache.write_text(json.dumps(found), encoding="utf-8")

    n_pos = sum(v.kind == "positive" for v in views)
    lines = ["# RampNet panorama model on perspective views", "",
             f"{n_pos} views aimed at a labeled ramp, {len(views) - n_pos} ramp-free views.",
             "For comparison, Mask2Former 'curb cut' found 45% at 0.13 extras/view and 68% at 0.41 extras/view.", ""]
    for name in VARIANTS:
        lines += [f"## {name}", "", "| score threshold | ramps found | extras / ramp-free view |", "|---|---|---|"]
        for t in THRESHOLDS:
            recall, extras = score(views, found[name], t)
            lines.append(f"| {t:.2f} | {recall:.0%} | {extras:.2f} |")
        lines.append("")

    lines += ["## Ramps found by distance (exact camera)", "", "| distance | views | " + " | ".join(f"score >= {t}" for t in (0.1, 0.2, 0.3)) + " |", "|---|---|---|---|---|"]
    for lo, hi in DISTANCE_BANDS:
        band = [v for v in views if v.kind == "positive" and v.distance_m and lo <= v.distance_m < hi]
        cells = [f"{score(views, found['exact camera'], t, band)[0]:.0%}" for t in (0.1, 0.2, 0.3)]
        lines.append(f"| {lo}-{hi} m | {len(band)} | " + " | ".join(cells) + " |")
    report = BENCH_DIR / "rampnet_probe.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
