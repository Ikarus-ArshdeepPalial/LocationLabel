from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np

from .models import Box

KERNEL = np.ones((3, 3), np.uint8)


@dataclass
class Instance:
    cls: str
    mask: np.ndarray
    confidence: float
    source: str

    @property
    def bbox(self) -> list[int]:
        ys, xs = np.nonzero(self.mask)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def centroid(self) -> list[float]:
        ys, xs = np.nonzero(self.mask)
        return [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]


def map_labels(id2label: dict[int, str], class_map: dict[str, list[str]]) -> tuple[dict[int, str], list[str]]:
    """Model class id -> canonical class, plus the canonical classes with no matching model label."""
    by_name = {name.lower(): model_id for model_id, name in id2label.items()}
    id_to_class, unmatched = {}, []
    for cls, names in class_map.items():
        ids = [by_name[n.lower()] for n in names if n.lower() in by_name]
        id_to_class.update({i: cls for i in ids})
        if not ids:
            unmatched.append(cls)
    return id_to_class, unmatched


def canonical_map(labels: np.ndarray, id_to_class: dict[int, str], classes: list[str]) -> np.ndarray:
    lookup = np.zeros(int(labels.max()) + 1, np.uint8)
    for model_id, cls in id_to_class.items():
        if model_id < len(lookup):
            lookup[model_id] = classes.index(cls)
    return lookup[labels]


def split_instances(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL)
    count, components, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    return [components == i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def box_mask(xyxy: tuple[float, ...], shape: tuple[int, ...]) -> np.ndarray:
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    mask = np.zeros(shape[:2], bool)
    mask[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1] = True
    return mask


def box_overlap(xyxy: tuple[float, ...], mask: np.ndarray) -> float:
    region = box_mask(xyxy, mask.shape)
    return float((region & mask).sum() / max(region.sum(), 1))


def split_by_confidence(boxes: list[Box], threshold: float) -> tuple[list[Box], list[Box]]:
    return [b for b in boxes if b.score >= threshold], [b for b in boxes if b.score < threshold]


def associate_pole(xyxy: tuple[float, ...], poles: list[np.ndarray]) -> int | None:
    """Index of the pole whose mask overlaps the sign or runs directly beneath it."""
    x1, y1, x2, y2 = xyxy
    margin = (x2 - x1) * 0.25
    columns = slice(max(int(x1 - margin), 0), int(x2 + margin) + 1)
    best, best_score = None, 0
    for i, pole in enumerate(poles):
        ys, _ = np.nonzero(pole)
        if ys.min() > y2 + (y2 - y1):
            continue  # starts well below the sign, so it's a different object
        score = int(pole[int(y1):, columns].sum())
        if score > best_score:
            best, best_score = i, score
    return best


def lowest_pixel(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    bottom = ys.max()
    return [int(round(xs[ys == bottom].mean())), int(bottom)]


def curb_line(instance: np.ndarray, road: np.ndarray, max_points: int = 64) -> list[list[int]]:
    """Road pixels touching the instance, as one point per image column (median row), left to right."""
    touching = road & cv2.dilate(instance.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ys, xs = np.nonzero(touching)
    if not len(xs):
        return []
    points = [[int(x), int(np.median(ys[xs == x]))] for x in np.unique(xs)]
    step = ceil(len(points) / max_points)
    return points[::step] if points[::step][-1] == points[-1] else points[::step] + [points[-1]]


def find_pad(classes: np.ndarray, border: np.ndarray, sidewalk_id: int, min_area_fraction: float = 0.01) -> np.ndarray | None:
    """A tactile or brick pad: a compact unlabeled patch that the sidewalk surrounds. Curb ramps carry one, so when the
    ramp detector finds nothing it still marks the ramp; the largest such patch is returned, or None."""
    height, width = classes.shape
    hole = ((classes == 0) & ~border).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
    ring_size = max(5, round(0.02 * max(height, width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_size + 1,) * 2)
    best, best_area = None, 0
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area_fraction * height * width or area <= best_area:
            continue
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue  # cut by the frame: not a whole pad
        mask = labels == i
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
        ring = (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & ~mask
        if hull_area <= 0 or area / hull_area < 0.8 or (classes[ring] == sidewalk_id).mean() < 0.5:
            continue
        best, best_area = mask, area
    return best
