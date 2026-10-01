"""Scores the ramp detector on RampNet's labeled Street View panoramas, rendered as ordinary perspective photos.

Positive views are aimed at a labeled ramp (a miss is a real miss). Views aimed away from every label are only
"ramp-free" by the inventory: driveways and unlisted ramps count as ramps for us, so detections there are reported
as extras to inspect, never as false positives.
"""

import json
import logging
import math
import random
from statistics import median
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DATA_DIR

logger = logging.getLogger(__name__)

BENCH_DIR = DATA_DIR / "rampbench"
DATASET = "projectsidewalk/rampnet-dataset"
CAMERA_HEIGHT_M = 2.5
VIEW_SIZE, VIEW_FOV, VIEW_PITCH = 640, 90.0, -12.0
DEPRESSION_RANGE_DEG = (8.0, 35.0)  # ~4-18 m from the camera on flat ground
HIT_RADIUS_PX = 60
NEGATIVE_MIN_YAW_GAP_DEG = 70
AREA_GRID = (0.0, 0.0005, 0.001, 0.003, 0.01)
CONFIDENCE_GRID = (0.0, 0.4, 0.55, 0.7)
ASPECT_GRID = (None, 12, 8, 6, 4)
SHAPE_SWEEP_AT = (0.0005, 0.4)  # min area, min confidence held fixed while the aspect cap varies


def view_rays(size: int, fov_deg: float, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """World direction (x east, y up, z north) of every pixel of a pinhole view with no roll."""
    f = (size / 2) / math.tan(math.radians(fov_deg) / 2)
    u, v = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    x, y, z = (u - size / 2) / f, -(v - size / 2) / f, np.ones_like(u)
    p, w = math.radians(pitch_deg), math.radians(yaw_deg)
    y2, z2 = y * math.cos(p) + z * math.sin(p), -y * math.sin(p) + z * math.cos(p)
    return np.stack([x * math.cos(w) + z2 * math.sin(w), y2, -x * math.sin(w) + z2 * math.cos(w)], axis=-1)


def render_view(pano: np.ndarray, yaw_deg: float, pitch_deg: float = VIEW_PITCH, size: int = VIEW_SIZE, fov_deg: float = VIEW_FOV) -> np.ndarray:
    d = view_rays(size, fov_deg, yaw_deg, pitch_deg)
    lon, lat = np.arctan2(d[..., 0], d[..., 2]), np.arctan2(d[..., 1], np.hypot(d[..., 0], d[..., 2]))
    h, w = pano.shape[:2]
    map_x = ((lon / (2 * np.pi) + 0.5) * w).astype(np.float32)
    map_y = ((0.5 - lat / np.pi) * h).astype(np.float32)
    return cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def project(u_norm: float, v_norm: float, yaw_deg: float, pitch_deg: float = VIEW_PITCH, size: int = VIEW_SIZE, fov_deg: float = VIEW_FOV) -> tuple[float, float] | None:
    """Pixel where a panorama point (normalized x, y) lands in a rendered view, or None if it is outside it."""
    lon, lat = (u_norm - 0.5) * 2 * math.pi, (0.5 - v_norm) * math.pi
    dx, dy, dz = math.sin(lon) * math.cos(lat), math.sin(lat), math.cos(lon) * math.cos(lat)
    w, p = math.radians(yaw_deg), math.radians(pitch_deg)
    x, z2 = dx * math.cos(w) - dz * math.sin(w), dx * math.sin(w) + dz * math.cos(w)
    y, z = dy * math.cos(p) - z2 * math.sin(p), dy * math.sin(p) + z2 * math.cos(p)
    if z <= 1e-6:
        return None
    f = (size / 2) / math.tan(math.radians(fov_deg) / 2)
    px, py = size / 2 + f * x / z, size / 2 - f * y / z
    return (px, py) if 0 <= px < size and 0 <= py < size else None


def depression_deg(v_norm: float) -> float:
    return (v_norm - 0.5) * 180


def yaw_of(u_norm: float) -> float:
    return (u_norm - 0.5) * 360


def yaw_gap(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


@dataclass
class ViewMeta:
    name: str
    kind: str  # "positive" or "negative"
    pano_id: str
    yaw: float
    gt_pixel: tuple[float, float] | None = None
    distance_m: float | None = None


def build_views(n_panos: int, seed: int) -> list[ViewMeta]:
    """Streams panoramas and renders the views to BENCH_DIR/views; skipped if a matching cache exists."""
    views_dir = BENCH_DIR / "views"
    index = BENCH_DIR / "views.json"
    if index.exists():
        cached = [ViewMeta(**{**m, "gt_pixel": tuple(m["gt_pixel"]) if m["gt_pixel"] else None}) for m in json.loads(index.read_text())]
        if len({m.pano_id for m in cached}) >= n_panos:
            logger.info("Using %d cached views from %d panoramas", len(cached), len({m.pano_id for m in cached}))
            return cached

    from datasets import load_dataset

    views_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    stream = load_dataset(DATASET, split="test", streaming=True).shuffle(seed=seed, buffer_size=50)
    views: list[ViewMeta] = []
    panos = 0
    for row in stream:
        points = row["curb_ramp_points_normalized"]
        usable = [(u, v) for u, v in points if DEPRESSION_RANGE_DEG[0] <= depression_deg(v) <= DEPRESSION_RANGE_DEG[1]]
        if not usable:
            continue
        pano = np.asarray(row["image"].convert("RGB"))
        pano_id = row["pano_id"]

        for i, (u, v) in enumerate(rng.sample(usable, min(2, len(usable)))):
            yaw = yaw_of(u)
            name = f"{pano_id}_pos{i}"
            cv2.imwrite(str(views_dir / f"{name}.jpg"), cv2.cvtColor(render_view(pano, yaw), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            distance = CAMERA_HEIGHT_M / math.tan(math.radians(depression_deg(v)))
            views.append(ViewMeta(name, "positive", pano_id, yaw, project(u, v, yaw), round(distance, 1)))

        label_yaws = [yaw_of(u) for u, _ in points]
        free = [y for y in np.arange(-180, 180, 10) if all(yaw_gap(y, ly) >= NEGATIVE_MIN_YAW_GAP_DEG for ly in label_yaws)]
        if free:
            yaw = float(rng.choice(free))
            name = f"{pano_id}_neg"
            cv2.imwrite(str(views_dir / f"{name}.jpg"), cv2.cvtColor(render_view(pano, yaw), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            views.append(ViewMeta(name, "negative", pano_id, yaw))

        panos += 1
        logger.info("Rendered panorama %d/%d", panos, n_panos)
        if panos >= n_panos:
            break

    index.write_text(json.dumps([m.__dict__ for m in views]), encoding="utf-8")
    return views


def hits(mask: np.ndarray, point: tuple[float, float], radius: int = HIT_RADIUS_PX) -> bool:
    x, y = point
    h, w = mask.shape
    x0, x1, y0, y1 = max(int(x - radius), 0), min(int(x + radius) + 1, w), max(int(y - radius), 0), min(int(y + radius) + 1, h)
    if x0 >= x1 or y0 >= y1:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return bool(mask[y0:y1, x0:x1][(xx - x) ** 2 + (yy - y) ** 2 <= radius**2].any())


def mask_aspect(mask: np.ndarray) -> float:
    """Length over width of the tightest rotated rectangle around the region; a thin strip along a curb is large."""
    ys, xs = np.nonzero(mask)
    (_, _), (w, h), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
    return max(w, h) / max(min(w, h), 1.0)


def score_views(views: list[ViewMeta], seg_size: int, defaults: tuple[float, float]) -> list[dict]:
    """Per view: every ramp candidate (area fraction, confidence, whether it hit the label). Saves example images."""
    from label_objs.config import CLASSES, CLASS_MAP, load_settings
    from label_objs.models import HFSegmentationModel, select_device
    from label_objs.postprocess import canonical_map, map_labels, split_instances

    settings = load_settings()
    segmenter = HFSegmentationModel(settings.seg_model, seg_size, select_device())
    id_to_class, _ = map_labels(segmenter.id2label, CLASS_MAP)
    examples_dir = BENCH_DIR / "examples" / str(seg_size)
    examples_dir.mkdir(parents=True, exist_ok=True)

    results, saved = [], {"miss": 0, "extra": 0}
    for view in views:
        bgr = cv2.imread(str(BENCH_DIR / "views" / f"{view.name}.jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        seg = segmenter.predict([rgb])[0]
        ramp_mask = canonical_map(seg.labels, id_to_class, CLASSES) == CLASSES.index("curb_ramp")
        candidates = []
        for mask in split_instances(ramp_mask, settings.min_area["curb_ramp"]):
            candidates.append({
                "area_frac": float(mask.sum()) / mask.size,
                "conf": float(seg.confidence[mask].mean()),
                "hit": bool(view.gt_pixel and hits(mask, view.gt_pixel)),
                "aspect": mask_aspect(mask),
                "mask": mask,
            })
        results.append({"view": view, "candidates": candidates})

        kept = [c for c in candidates if c["area_frac"] >= defaults[0] and c["conf"] >= defaults[1]]
        kind = "miss" if view.kind == "positive" and not any(c["hit"] for c in kept) else "extra" if view.kind == "negative" and kept else None
        if kind and saved[kind] < 10:
            canvas = bgr.copy()
            for c in kept:
                contours, _ = cv2.findContours(c["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, (255, 255, 0), 2)
            if view.gt_pixel:
                cv2.circle(canvas, (int(view.gt_pixel[0]), int(view.gt_pixel[1])), HIT_RADIUS_PX, (0, 0, 255), 2)
            cv2.imwrite(str(examples_dir / f"{kind}_{view.name}.jpg"), canvas)
            saved[kind] += 1
    return results


def evaluate(results: list[dict], min_area: float, min_conf: float, max_aspect: float | None) -> dict:
    positives = [r for r in results if r["view"].kind == "positive"]
    negatives = [r for r in results if r["view"].kind == "negative"]

    def keep(r):
        return [c for c in r["candidates"] if c["area_frac"] >= min_area and c["conf"] >= min_conf
                and (max_aspect is None or c["aspect"] <= max_aspect)]

    return {
        "recall": sum(any(c["hit"] for c in keep(r)) for r in positives) / max(len(positives), 1),
        "extras_per_free_view": sum(len(keep(r)) for r in negatives) / max(len(negatives), 1),
    }


def load_or_score(views: list[ViewMeta], seg_size: int, defaults: tuple[float, float], rescore: bool) -> list[dict]:
    """Candidates per view for one segmentation size, cached as JSON so threshold sweeps don't re-run the model."""
    path = BENCH_DIR / f"candidates_{seg_size}.json"
    if path.exists() and not rescore:
        by_name = {v.name: v for v in views}
        return [{"view": by_name[r["view"]], "candidates": r["candidates"]} for r in json.loads(path.read_text())]

    results = score_views(views, seg_size, defaults)
    path.write_text(json.dumps([
        {"view": r["view"].name, "candidates": [{k: v for k, v in c.items() if k != "mask"} for c in r["candidates"]]}
        for r in results
    ]), encoding="utf-8")
    return results


def run_rampbench(n_panos: int = 60, seg_sizes: tuple[int, ...] = (640, 1024), seed: int = 0, rescore: bool = False) -> Path:
    from label_objs.config import Settings

    views = build_views(n_panos, seed)
    n_pos = sum(v.kind == "positive" for v in views)
    lines = [
        "# Ramp detector benchmark (RampNet panoramas rendered as 90° photos)", "",
        f"{n_pos} views aimed at a labeled ramp (4-18 m away), {len(views) - n_pos} ramp-free views.",
        f"A ramp counts as found if a detected ramp region comes within {HIT_RADIUS_PX}px of its label.",
        "Extras are detections in views without a labeled ramp; some are real driveways or unlisted ramps.", "",
    ]
    defaults = (Settings.ramp_min_area_fraction, Settings.ramp_min_confidence)
    for seg_size in seg_sizes:
        results = load_or_score(views, seg_size, defaults, rescore)
        lines += [f"## Segmentation input {seg_size}px", "", "| min area | min conf | recall | extras / free view |", "|---|---|---|---|"]
        for area in AREA_GRID:
            for conf in CONFIDENCE_GRID:
                r = evaluate(results, area, conf, None)
                mark = " **(current)**" if (area, conf) == defaults else ""
                lines.append(f"| {area:.2%} | {conf:.2f} | {r['recall']:.0%}{mark} | {r['extras_per_free_view']:.2f} |")

        hit_aspects = [c["aspect"] for r in results for c in r["candidates"] if c["hit"]]
        free_aspects = [c["aspect"] for r in results if r["view"].kind == "negative" for c in r["candidates"]]
        lines += ["", f"Shape filter at min area {SHAPE_SWEEP_AT[0]:.2%}, min conf {SHAPE_SWEEP_AT[1]:.2f} "
                      f"(median length/width: {median(hit_aspects):.1f} on labeled ramps, n={len(hit_aspects)}; "
                      f"{median(free_aspects):.1f} in ramp-free views, n={len(free_aspects)}):", "",
                  "| max length/width | recall | extras / free view |", "|---|---|---|"]
        for aspect in ASPECT_GRID:
            r = evaluate(results, *SHAPE_SWEEP_AT, aspect)
            lines.append(f"| {'no limit' if aspect is None else aspect} | {r['recall']:.0%} | {r['extras_per_free_view']:.2f} |")
        lines.append("")

    report = BENCH_DIR / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Benchmark report: %s", report)
    return report
