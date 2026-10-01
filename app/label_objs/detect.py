import hashlib
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.cache import JsonCache
from common.progress import progress
from common.record import load_record, record_path, update_record

from .annotate import annotate, contact_sheet
from .config import CLASS_MAP, CLASSES, COLORS, CURB_LINE_CLASSES, ROAD_SIDE_CLASSES, TARGET_CLASSES, TARGETS, Settings
from .models import Box, BoxDetector, RampPoint, RampPointDetector, RegionDetector, Segmentation, SegmentationModel
from .postprocess import (
    Instance,
    associate_pole,
    box_mask,
    box_overlap,
    canonical_map,
    curb_line,
    find_pad,
    lowest_pixel,
    map_labels,
    split_by_confidence,
    split_instances,
)

logger = logging.getLogger(__name__)


class DetectionStage:
    def __init__(
        self,
        settings: Settings,
        segmenter: SegmentationModel,
        sign_detector: BoxDetector | None,
        cache: JsonCache,
        ramp_detector: RampPointDetector | None,
        force: bool = False,
        driveway_detector: RegionDetector | None = None,
        targets: tuple[str, ...] = TARGETS,
    ) -> None:
        self._settings = settings
        self._segmenter = segmenter
        self._sign_detector = sign_detector
        self._cache = cache
        self._ramp_detector = ramp_detector
        self._driveway_detector = driveway_detector
        self._targets = targets
        self._force = force
        self._id_to_class, unmatched = map_labels(segmenter.id2label, CLASS_MAP)
        for cls in unmatched:
            logger.warning("Canonical class %r has no match in %s labels", cls, segmenter.version)

    @property
    def _signature(self) -> str:
        parts = [
            self._segmenter.version,
            self._sign_detector.version if self._sign_detector else None,
            self._settings.seg_input_size,
            self._settings.stop_sign_threshold,
            self._ramp_detector.version if self._ramp_detector else None,
            self._driveway_detector.version if self._driveway_detector else None,
            self._settings.driveway_prompt,
            self._settings.driveway_min_score,
            self._settings.driveway_min_area_fraction,
            sorted(self._targets),
            self._settings.ramp_min_score,
            self._settings.ramp_min_area_fraction,
            self._settings.ramp_min_confidence,
            self._settings.ramp_region_radius,
            self._settings.min_area,
        ]
        return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]

    def run(self, address: str) -> str:
        path = record_path(self._settings.output_dir, address)
        if not path.exists():
            logger.error("No record for %r at %s, run the geocode and streetview stages first", address, path)
            return "error"

        try:
            section = self._process(load_record(path), path.parent)
        except Exception as exc:
            logger.exception("Detection stage failed for %r", address)
            section = {"status": "error", "message": f"Unexpected error: {exc}"}

        update_record(path, {"detections": section})
        log = logger.info if section["status"] == "ok" else logger.warning
        log("%r: detections %s - %s", address, section["status"], section.get("message", ""))
        return section["status"]

    def _process(self, record: dict[str, Any], address_dir: Path) -> dict[str, Any]:
        images = (record.get("streetview") or {}).get("images") or []
        if not images:
            return {"status": "error", "message": "No Street View images, run the streetview stage first"}

        for folder in ("masks", "annotated"):
            (address_dir / folder).mkdir(exist_ok=True)
        self._write_legend(address_dir / "masks" / "labels.json")

        entries: dict[str, dict] = {}
        failed: list[dict] = []
        pending: list[tuple[str, np.ndarray, str, dict]] = []
        for meta in images:
            name = meta["filename"]
            try:
                data = (address_dir / "images" / name).read_bytes()
                key = f"{hashlib.sha256(data).hexdigest()}:{self._signature}:{meta.get('group', 'main')}"
                cached = None if self._force else self._cache.get(key)
                if cached and (address_dir / cached["annotated"]).exists():
                    logger.debug("Cached detections for %s", name)
                    entries[name] = cached
                    continue
                pending.append((name, decode_rgb(data), key, meta))
            except Exception as exc:
                logger.warning("Could not read %s: %s", name, exc)
                failed.append({"image": name, "error": str(exc)})

        size = self._settings.batch_size
        with progress(total=len(pending), desc="  labeling") as bar:
            for start in range(0, len(pending), size):
                self._label_batch(pending[start:start + size], address_dir, entries, failed)
                bar.update(len(pending[start:start + size]))

        per_image = [entries[m["filename"]] for m in images if m["filename"] in entries]
        if per_image:
            sheet = [(address_dir / e["annotated"], f"{e['image'][:24]} {describe(e)}") for e in per_image]
            contact_sheet(sheet, address_dir / "contact_sheet_annotated.jpg")
        return self._section(per_image, failed)

    def _label_batch(self, batch: list[tuple[str, np.ndarray, str, dict]], address_dir: Path, entries: dict, failed: list) -> None:
        try:
            rgbs = [rgb for _, rgb, _, _ in batch]
            segmentations = self._segmenter.predict(rgbs)
            sign_boxes = self._sign_detector.detect(rgbs) if self._sign_detector else [[] for _ in rgbs]
        except Exception as exc:
            logger.exception("Inference failed for %s", [name for name, _, _, _ in batch])
            failed += [{"image": name, "error": str(exc)} for name, _, _, _ in batch]
            return

        for (name, rgb, key, meta), segmentation, boxes in zip(batch, segmentations, sign_boxes):
            try:
                entries[name] = self._label_image(name, rgb, segmentation, boxes, meta, address_dir)
                self._cache.put(key, entries[name])
            except Exception as exc:
                logger.exception("Labeling failed for %s", name)
                failed.append({"image": name, "error": str(exc)})

    def _ramp_instances(self, rgb: np.ndarray, meta: dict, regions: list[Instance]) -> list[tuple[Instance, dict]]:
        """Curb ramps from RampNet points and from big, confident segmented curb-cut regions.

        A point takes the nearest segmented region under it (even a weak one) as the area to measure; a region with
        no point still counts if it is big and confident enough.
        """
        k = meta["intrinsics"]
        height, width = rgb.shape[:2]
        radius = self._settings.ramp_region_radius * math.hypot(width, height)
        points = [p for p in self._ramp_detector.detect(rgb, k["fx"], k["cx"], k["cy"], meta["pitch"]) if p.score >= self._settings.ramp_min_score]

        found, used = [], set()
        for point in sorted(points, key=lambda p: -p.score):
            near = [(distance_to_mask(r.mask, point), i) for i, r in enumerate(regions)]
            distance, i = min(near, default=(float("inf"), None))
            if i is not None and distance <= radius:
                mask, source = regions[i].mask, "segmentation"
                used.add(i)
            else:
                mask, source = np.zeros((height, width), bool), "point"
                cv2.circle(mask.view(np.uint8), (round(point.x), round(point.y)), max(6, round(0.01 * math.hypot(width, height))), 1, -1)
            found.append((Instance("curb_ramp", mask, point.score, self._ramp_detector.name), {"point": [round(point.x, 1), round(point.y, 1)], "mask_source": source}))

        for i, region in enumerate(regions):
            if i not in used and region.area >= self._settings.ramp_min_area_fraction * height * width and region.confidence >= self._settings.ramp_min_confidence:
                found.append((region, {"mask_source": "segmentation"}))
        return found

    def _wanted(self, cls: str) -> bool:
        """Poles and curbs are reported only when asked for; every other segmented class is context that later stages use."""
        gated = {"pole": "poles", "curb": "curbs"}
        return cls not in gated or gated[cls] in self._targets

    def _driveway_instances(self, rgb: np.ndarray, classes: np.ndarray, border: np.ndarray) -> list[tuple[Instance, dict]]:
        """Driveways as ramps: every driveway meets the street at an apron."""
        s = self._settings
        height, width = rgb.shape[:2]
        road = classes == CLASSES.index("road")
        grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(3, round(0.01 * math.hypot(width, height))) + 1,) * 2)

        found = []
        for region in sorted(self._driveway_detector.detect(rgb, s.driveway_prompt, s.driveway_min_score), key=lambda r: -r.score):
            mask = largest_part(cut_thin_strips(region.mask & ~border & ~road))
            if region.score < s.driveway_min_score or mask.mean() < s.driveway_min_area_fraction:
                continue
            on_border = mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
            if not (on_border or (cv2.dilate(mask.astype(np.uint8), grow) > 0)[road].any()):
                continue  # floating in the frame and away from the road: a parking lot or a far-off driveway
            if any((mask & other.mask).sum() >= 0.5 * min(mask.sum(), other.mask.sum()) for other, _ in found):
                continue  # the same driveway found again at a lower score
            found.append((
                Instance("curb_ramp", mask, region.score, self._driveway_detector.name),
                {"mask_source": "driveway", "kind": "driveway", "outline": outline(mask)},
            ))
        return found

    def _label_image(self, name: str, rgb: np.ndarray, seg: Segmentation, sign_boxes: list[Box], meta: dict, address_dir: Path) -> dict:
        stem = Path(name).stem
        classes = canonical_map(seg.labels, self._id_to_class, CLASSES)
        border = rotation_border(rgb)
        classes[border] = 0
        instances = [
            Instance(cls, mask, float(seg.confidence[mask].mean()), self._segmenter.name)
            for cls in CLASS_MAP
            for mask in split_instances(classes == CLASSES.index(cls), self._settings.min_area[cls])
        ]

        regions = [i for i in instances if i.cls == "curb_ramp"]
        instances = [i for i in instances if i.cls != "curb_ramp" and self._wanted(i.cls)]
        ramps = self._ramp_instances(rgb, meta, regions) if self._ramp_detector else []
        if meta.get("group") == "ramp":
            # The photos are known to show a ramp, so a detector miss is not the end: its tactile pad marks it.
            pad = find_pad(classes, border, CLASSES.index("sidewalk"))
            if pad is not None:
                ramps = [(Instance("curb_ramp", pad, 0.5, "tactile_pad"), {"mask_source": "segmentation", "kind": "ramp", "region": "tactile_pad"})]

        kept, dropped = split_by_confidence(sign_boxes, self._settings.stop_sign_threshold)
        for box in dropped:
            logger.info("Dropped stop sign in %s with confidence %.2f (< %.2f)", name, box.score, self._settings.stop_sign_threshold)

        counters: Counter[str] = Counter()
        records: list[dict] = []
        semantic = np.zeros(rgb.shape[:2], np.uint8)

        def add(inst: Instance, paint: bool = True, **extra: Any) -> dict:
            instance_id = f"{stem}_{inst.cls}_{counters[inst.cls]}"
            counters[inst.cls] += 1
            mask_path = f"masks/{instance_id}.png"
            cv2.imwrite(str(address_dir / mask_path), inst.mask.astype(np.uint8) * 255)
            if paint:
                semantic[inst.mask] = CLASSES.index(inst.cls)
            entry = {
                "instance_id": instance_id,
                "class": inst.cls,
                "source": inst.source,
                "confidence": round(inst.confidence, 4),
                "bbox": inst.bbox,
                "area_px": inst.area,
                "centroid": inst.centroid,
                "mask": mask_path,
                **extra,
            }
            records.append(entry)
            return entry

        road = np.isin(classes, [CLASSES.index(c) for c in ROAD_SIDE_CLASSES])
        pole_entries = []
        for inst in instances:
            extra = {"curb_line": curb_line(inst.mask, road)} if inst.cls in CURB_LINE_CLASSES else {}
            entry = add(inst, **extra)
            if inst.cls == "pole":
                pole_entries.append((entry, inst.mask))

        driveways = self._driveway_instances(rgb, classes, border) if self._driveway_detector else []
        claimed: set[int] = set()
        for k, (ramp, extra) in enumerate(ramps):
            # A segmented ramp region inside a driveway is its apron: keep it for measuring and tag it as the driveway.
            inside = next((d for d, (drive, _) in enumerate(driveways) if extra["mask_source"] == "segmentation" and (drive.mask & ramp.mask).sum() >= 0.5 * ramp.area), None)
            if inside is not None:
                claimed.add(inside)
                ramps[k] = (ramp, {**extra, "kind": "driveway", "outline": driveways[inside][1]["outline"]})

        for inst, extra in ramps:
            has_region = extra["mask_source"] == "segmentation"
            add(inst, paint=has_region, **extra, **({"curb_line": curb_line(inst.mask, road)} if has_region else {}))

        for d, (inst, extra) in enumerate(driveways):
            if d in claimed:
                continue
            add(inst, paint=False, curb_line=curb_line(inst.mask, road), **extra)

        traffic_signs = classes == CLASSES.index("traffic_sign")
        for box in kept:
            region = box_mask(box.xyxy, rgb.shape)
            sign_pixels = region & traffic_signs
            pole_index = associate_pole(box.xyxy, [mask for _, mask in pole_entries])
            pole = None
            if pole_index is not None:
                pole_entry, pole_mask = pole_entries[pole_index]
                pole = {"instance_id": pole_entry["instance_id"], "base_pixel": lowest_pixel(pole_mask)}
            x1, y1, x2, y2 = box.xyxy
            add(
                Instance("stop_sign", sign_pixels if sign_pixels.any() else region, box.score, self._sign_detector.name),
                bbox=[round(v) for v in box.xyxy],
                bottom_center=[round((x1 + x2) / 2, 1), round(y2, 1)],
                traffic_sign_overlap=round(box_overlap(box.xyxy, traffic_signs), 3),
                pole=pole,
            )

        cv2.imwrite(str(address_dir / "masks" / f"{stem}_semantic.png"), semantic)
        annotated = annotate(rgb, semantic, records)
        cv2.imwrite(str(address_dir / "annotated" / f"{stem}.jpg"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        return {
            "image": name,
            "semantic": f"masks/{stem}_semantic.png",
            "annotated": f"annotated/{stem}.jpg",
            "instances": records,
        }

    def _write_legend(self, path: Path) -> None:
        legend = {
            "format": "uint8 PNG, pixel value = class index, 0 = unlabeled",
            "classes": dict(enumerate(CLASSES)),
            "colors": COLORS,
            "segmentation_model": self._segmenter.version,
        }
        path.write_text(json.dumps(legend, indent=2), encoding="utf-8")

    def _section(self, per_image: list[dict], failed: list[dict]) -> dict[str, Any]:
        def images_with(cls: str) -> int:
            return sum(any(i["class"] == cls for i in e["instances"]) for e in per_image)

        counts = Counter(i["class"] for e in per_image for i in e["instances"])
        summary = {
            "stop_sign_found": counts["stop_sign"] > 0,
            "curb_ramp_found": counts["curb_ramp"] > 0,
            "driveway_found": any(i.get("kind") == "driveway" for e in per_image for i in e["instances"]),
            "num_images_with_stop_sign": images_with("stop_sign"),
            "num_images_with_ramp": images_with("curb_ramp"),
            "num_images_processed": len(per_image),
            "num_images_failed": len(failed),
            "instance_counts": dict(counts),
        }
        if not per_image:
            status = "error"
        elif failed:
            status = "partial"
        elif not any(counts[c] for c in TARGET_CLASSES):
            status = "no_detections"
        else:
            status = "ok"

        models = {"segmentation": self._segmenter, "stop_sign": self._sign_detector, "ramps": self._ramp_detector, "driveways": self._driveway_detector}
        return {
            "status": status,
            "message": f"{len(per_image)} images labeled, {len(failed)} failed",
            "models": {role: {"name": m.name, "version": m.version} for role, m in models.items() if m},
            "per_image": per_image,
            "failed_images": failed,
            "summary": summary,
            "contact_sheet": "contact_sheet_annotated.jpg" if per_image else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def rotation_border(rgb: np.ndarray) -> np.ndarray:
    """Black wedges that straightening a tilted photo leaves at its edges; they are not scene content."""
    dark = (rgb.max(axis=2) <= 6).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    on_edge = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
    keep = [i for i in on_edge if i and stats[i, cv2.CC_STAT_AREA] > 0.002 * dark.size]
    return cv2.dilate(np.isin(labels, keep).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0


def cut_thin_strips(mask: np.ndarray) -> np.ndarray:
    """Removes narrow arms (a sidewalk strip the segmenter fused to the driveway) unless that would lose over half the mask."""
    size = max(3, round(0.03 * min(mask.shape)))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0
    return opened if opened.sum() >= 0.5 * mask.sum() else mask


def largest_part(mask: np.ndarray) -> np.ndarray:
    """The biggest connected piece of a mask; the text-prompted segmenter often leaks into neighbouring surfaces."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def outline(mask: np.ndarray) -> list[list[int]]:
    """A simplified polygon around the largest part of a mask, for drawing."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    return cv2.approxPolyDP(largest, 0.002 * cv2.arcLength(largest, True), True)[:, 0].tolist()


def distance_to_mask(mask: np.ndarray, point: RampPoint) -> float:
    ys, xs = np.nonzero(mask)
    return float(np.min(np.hypot(xs - point.x, ys - point.y))) if len(xs) else float("inf")


def decode_rgb(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not a decodable image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def describe(entry: dict) -> str:
    counts = Counter(i["class"] for i in entry["instances"])
    return f"sign:{counts['stop_sign']} ramp:{counts['curb_ramp']} curb:{counts['curb']}"
