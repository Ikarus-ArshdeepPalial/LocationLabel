from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Crop:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def apply(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        return image[self.top:height - self.bottom, self.left:width - self.right]

    def union(self, other: "Crop") -> "Crop":
        return Crop(*(max(a, b) for a, b in zip(self.as_tuple(), other.as_tuple())))

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.top, self.bottom, self.left, self.right


def uniform_run(profile: np.ndarray, threshold: float, limit: int) -> int:
    """Number of leading entries below threshold, capped at limit."""
    above = np.nonzero(profile[:limit] >= threshold)[0]
    return int(above[0]) if len(above) else limit


def detect_ui_bars(image: np.ndarray, max_fraction: float = 0.2, threshold: float = 6.0) -> Crop:
    """Flat-coloured bars along the image edges (browser/app chrome), found as rows/columns with almost no texture."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    row_std, col_std = gray.std(axis=1), gray.std(axis=0)
    max_rows, max_cols = int(height * max_fraction), int(width * max_fraction)
    return Crop(
        top=uniform_run(row_std, threshold, max_rows),
        bottom=uniform_run(row_std[::-1], threshold, max_rows),
        left=uniform_run(col_std, threshold, max_cols),
        right=uniform_run(col_std[::-1], threshold, max_cols),
    )


def margin_crop(margins: dict[str, float], width: int, height: int) -> Crop:
    return Crop(
        top=round(margins.get("top", 0) * height),
        bottom=round(margins.get("bottom", 0) * height),
        left=round(margins.get("left", 0) * width),
        right=round(margins.get("right", 0) * width),
    )
