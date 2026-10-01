from math import ceil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import CLASSES, COLORS, UNLABELED_IN_ANNOTATION

ALPHA = 0.45
THUMB = 320
LABEL_HEIGHT = 22
COLUMNS = 4


def annotate(image: np.ndarray, semantic: np.ndarray, instances: list[dict[str, Any]]) -> np.ndarray:
    """RGB image with a class-coloured overlay, stop-sign boxes, ramp points, curb lines, pole bases and labels."""
    overlay = image.copy()
    for index, cls in enumerate(CLASSES):
        if cls in COLORS:
            overlay[semantic == index] = COLORS[cls]
    out = cv2.addWeighted(overlay, ALPHA, image, 1 - ALPHA, 0)

    for inst in instances:
        color = COLORS[inst["class"]]
        x1, y1, x2, y2 = inst["bbox"]
        if inst["class"] == "stop_sign":
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if inst["pole"]:
                cv2.circle(out, tuple(inst["pole"]["base_pixel"]), 5, COLORS["pole"], -1)
        if inst.get("outline"):
            cv2.polylines(out, [np.array(inst["outline"], np.int32)], True, color, max(2, image.shape[1] // 1000))
        if inst.get("point"):
            cv2.circle(out, tuple(int(v) for v in inst["point"]), max(12, image.shape[1] // 90), color, 3)
        if inst.get("curb_line"):
            cv2.polylines(out, [np.array(inst["curb_line"], np.int32)], False, color, 2)
        if inst["class"] not in UNLABELED_IN_ANNOTATION:
            text = f"{inst.get('kind', inst['class'])} {inst['confidence']:.2f}"
            cv2.putText(out, text, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(out, text, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def contact_sheet(images: list[tuple[Path, str]], out_path: Path) -> None:
    columns = min(COLUMNS, len(images))
    rows = ceil(len(images) / columns)
    sheet = np.full((rows * (THUMB + LABEL_HEIGHT), columns * THUMB, 3), 255, np.uint8)
    for i, (path, label) in enumerate(images):
        x, y = (i % columns) * THUMB, (i // columns) * (THUMB + LABEL_HEIGHT)
        sheet[y + LABEL_HEIGHT:y + LABEL_HEIGHT + THUMB, x:x + THUMB] = cv2.resize(cv2.imread(str(path)), (THUMB, THUMB))
        cv2.putText(sheet, label, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
