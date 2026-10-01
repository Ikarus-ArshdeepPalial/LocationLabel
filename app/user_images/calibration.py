"""Single-image camera calibration: focal length, pitch and roll when no metadata provides them."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from math import asin, atan2, degrees, hypot

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Calibration:
    focal_px: float | None
    pitch_deg: float
    roll_deg: float


def angles_from_up(up: np.ndarray) -> tuple[float, float]:
    """(pitch, roll) in degrees from the world-up direction in camera coordinates (x right, y down, z forward).

    Pitch is positive when looking up (Street View convention). Roll is how far image "up" is tilted clockwise;
    rotating the image counter-clockwise by it makes vertical lines vertical.
    """
    up = up / np.linalg.norm(up)
    return degrees(asin(np.clip(up[2], -1, 1))), degrees(atan2(up[0], -up[1]))


class Calibrator(ABC):
    name: str

    @abstractmethod
    def estimate(self, image: np.ndarray, focal_px: float | None) -> Calibration | None:
        """Estimate from an RGB image; focal_px is a known focal length or None. None when estimation fails."""


class GeoCalibCalibrator(Calibrator):
    """GeoCalib (Veicht et al., ECCV 2024): learned focal length and gravity direction; CPU or GPU."""

    name = "geocalib"

    @cached_property
    def _loaded(self):
        import torch
        from geocalib import GeoCalib

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("Loading GeoCalib on %s", device)
        return GeoCalib().to(device), device

    def estimate(self, image: np.ndarray, focal_px: float | None) -> Calibration | None:
        import torch

        model, device = self._loaded
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).to(device)
        if focal_px:
            result = model.calibrate(tensor, priors={"focal": torch.tensor([focal_px], device=device)})
        else:
            result = model.calibrate(tensor)
        # GeoCalib's "gravity" vector is the camera-frame up direction: (0, -1, 0) for a level camera.
        pitch, roll = angles_from_up(result["gravity"].vec3d.reshape(-1).cpu().numpy())
        return Calibration(float(result["camera"].f.reshape(-1)[0]), pitch, roll)


class LineCalibrator(Calibrator):
    """Pitch and roll from the vanishing point of near-vertical line segments; focal length is not estimated."""

    name = "lines"

    def __init__(
        self, default_fov_deg: float, max_tilt_deg: float = 25, min_lines: int = 6, inlier_deg: float = 1.5, iterations: int = 500
    ) -> None:
        self._default_fov = default_fov_deg
        self._max_tilt = max_tilt_deg
        self._min_lines = min_lines
        self._min_cos = np.cos(np.radians(inlier_deg))
        self._iterations = iterations

    def estimate(self, image: np.ndarray, focal_px: float | None) -> Calibration | None:
        height, width = image.shape[:2]
        focal = focal_px or width / 2 / np.tan(np.radians(self._default_fov) / 2)
        segments = cv2.createLineSegmentDetector().detect(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))[0]
        if segments is None:
            return None

        segments = segments.reshape(-1, 4)
        dx, dy = segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]
        long_enough = np.hypot(dx, dy) > 0.04 * hypot(width, height)
        near_vertical = np.degrees(np.arctan2(np.abs(dx), np.abs(dy))) < self._max_tilt
        segments = segments[long_enough & near_vertical]
        if len(segments) < self._min_lines:
            return None

        # Each segment is a line through two normalized image points; the vanishing point is their common point.
        ends = segments.reshape(-1, 2, 2)
        rays = np.concatenate([(ends - [width / 2, height / 2]) / focal, np.ones((len(ends), 2, 1))], axis=-1)
        lines = np.cross(rays[:, 0], rays[:, 1])
        lines /= np.linalg.norm(lines, axis=1, keepdims=True)
        inliers = self._consensus(lines, rays[:, :, :2], np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1))
        if inliers.sum() < self._min_lines:
            return None

        up = np.linalg.svd(lines[inliers])[2][-1]
        up = up if up[1] < 0 else -up
        pitch, roll = angles_from_up(up)
        return Calibration(None, pitch, roll)

    def _consensus(self, lines: np.ndarray, ends: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """RANSAC: the vanishing point that the most (length-weighted) segments point at, ignoring leaning clutter."""
        directions = ends[:, 1] - ends[:, 0]
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        midpoints = ends.mean(axis=1)
        rng = np.random.default_rng(0)
        best, best_score = np.zeros(len(lines), bool), 0.0
        for _ in range(self._iterations):
            vp = np.cross(*lines[rng.choice(len(lines), 2, replace=False)])
            toward = vp[:2] - midpoints * vp[2]  # direction from each midpoint to the (possibly infinite) point
            norms = np.linalg.norm(toward, axis=1)
            if not norms.all():
                continue
            inliers = np.abs((directions * toward).sum(axis=1)) / norms >= self._min_cos
            score = lengths[inliers].sum()
            if score > best_score:
                best, best_score = inliers, score
        return best


def build_calibrators(names: tuple[str, ...], default_fov_deg: float) -> list[Calibrator]:
    available = {"geocalib": GeoCalibCalibrator, "lines": lambda: LineCalibrator(default_fov_deg)}
    return [available[name]() for name in names]


def calibrate(calibrators: list[Calibrator], image: np.ndarray, focal_px: float | None) -> tuple[Calibration, str] | None:
    """First successful calibration in priority order, and which calibrator produced it."""
    for calibrator in calibrators:
        try:
            result = calibrator.estimate(image, focal_px)
        except Exception as exc:
            logger.warning("Calibrator %s failed: %s", calibrator.name, exc)
            continue
        if result is not None:
            return result, calibrator.name
    return None
