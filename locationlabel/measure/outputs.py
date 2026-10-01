from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from label_objs.config import CLASSES, COLORS

from .geometry import Camera

PURPLE = (142, 68, 214)  # measured value
GREY = (58, 54, 54)      # default value (not measured confidently)
RED = (204, 45, 62)      # measured and confidently over a legal limit
WHITE = (255, 255, 255)
SOFT_WHITE = (228, 228, 232)

BOLD_FONTS = ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")
REGULAR_FONTS = ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")


@dataclass
class Drawing:
    points: list[list[float]]  # two or more points draw a measurement line; one point only anchors the label
    value: str                 # large text, e.g. "218 cm"
    detail: str = ""           # small second line, e.g. "±6 cm · 72%"
    color: tuple[int, int, int] = PURPLE
    outline: list[list[int]] | None = None  # closed polygon around a whole driveway


@lru_cache
def font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    for name in BOLD_FONTS if bold else REGULAR_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def draw_measurements(image: np.ndarray, drawings: list[Drawing]) -> np.ndarray:
    """Clean photo with thick measurement lines and solid label boxes, sized to the image resolution."""
    height, width = image.shape[:2]
    scale = max(width, height) / 1000
    canvas = image.copy()
    for d in drawings:
        if d.outline:
            cv2.polylines(canvas, [np.array(d.outline, np.int32)], True, d.color, max(1, round(1.2 * scale)), cv2.LINE_AA)
        if len(d.points) >= 2 and not d.outline:  # an outlined driveway needs no line through it
            points = np.round(np.array(d.points)).astype(np.int32)
            cv2.polylines(canvas, [points], False, d.color, max(2, round(2.5 * scale)), cv2.LINE_AA)

    base = Image.fromarray(canvas).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    big, small = font(max(15, round(22 * scale)), True), font(max(10, round(11 * scale)), False)
    pad, gap = max(4, round(7 * scale)), max(6, round(10 * scale))
    placed: list[tuple[int, int, int, int]] = []

    for d in drawings:
        value_box = draw.textbbox((0, 0), d.value, font=big)
        detail_box = draw.textbbox((0, 0), d.detail, font=small) if d.detail else (0, 0, 0, 0)
        box_w = max(value_box[2] - value_box[0], detail_box[2] - detail_box[0]) + 2 * pad
        box_h = (value_box[3] - value_box[1]) + (detail_box[3] - detail_box[1] + pad // 2 if d.detail else 0) + 2 * pad

        xs, ys = [p[0] for p in d.points], [p[1] for p in d.points]
        x = max(xs) + gap if max(xs) + gap + box_w <= width else min(xs) - gap - box_w
        y = (min(ys) + max(ys)) / 2 - box_h / 2
        x, y = int(np.clip(x, 0, width - box_w)), int(np.clip(y, 0, height - box_h))
        while any(overlaps((x, y, x + box_w, y + box_h), other, gap // 2) for other in placed) and y + 2 * box_h < height:
            y += box_h + gap // 2
        placed.append((x, y, x + box_w, y + box_h))

        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=max(2, round(4 * scale)), fill=(*d.color, 235))
        draw.text((x + pad - value_box[0], y + pad - value_box[1]), d.value, font=big, fill=WHITE)
        if d.detail:
            top = y + pad + (value_box[3] - value_box[1]) + pad // 2
            draw.text((x + pad - detail_box[0], top - detail_box[1]), d.detail, font=small, fill=SOFT_WHITE)

    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def mask_outline(mask: np.ndarray) -> list[list[int]]:
    """A simplified polygon around the largest part of a mask, for drawing."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    return cv2.approxPolyDP(largest, 0.002 * cv2.arcLength(largest, True), True)[:, 0].tolist()


def ramp_profile(mask: np.ndarray, curb_line: list[list[int]]) -> list[list[float]]:
    """Line up the ramp: from where it meets the road to the middle of its far (top) edge."""
    ys, xs = np.nonzero(mask)
    top_rows = ys <= ys.min() + max(2, (ys.max() - ys.min()) // 10)
    top = [float(xs[top_rows].mean()), float(ys.min())]
    if curb_line:
        start = curb_line[len(curb_line) // 2]
        return [[float(start[0]), float(start[1])], top]
    bottom_rows = ys >= ys.max() - max(2, (ys.max() - ys.min()) // 10)
    return [[float(xs[bottom_rows].mean()), float(ys.max())], top]


def write_ply(path: Path, camera: Camera, depth: np.ndarray, semantic: np.ndarray, image: np.ndarray, stride: int = 2) -> None:
    """Binary PLY of the depth point cloud in local ENU metres; labeled pixels use their class colour."""
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride].reshape(2, -1)
    points = camera.points_from_depth(u, v, depth[v, u]).astype(np.float32)
    palette = np.array([COLORS.get(c, (0, 0, 0)) for c in CLASSES], np.uint8)
    classes = semantic[v, u]
    colors = np.where(classes[:, None] > 0, palette[classes], image[v, u])

    vertices = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, axis in enumerate("xyz"):
        vertices[axis] = points[:, i]
    for i, channel in enumerate(("red", "green", "blue")):
        vertices[channel] = colors[:, i]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)
