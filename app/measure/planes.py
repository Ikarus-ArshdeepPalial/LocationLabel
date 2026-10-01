from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    normal: np.ndarray
    offset: float
    rms: float
    inlier_ratio: float
    n_points: int

    def z_at(self, xy: np.ndarray) -> np.ndarray:
        n = self.normal
        return -(xy @ n[:2] + self.offset) / n[2]

    def slope_along(self, direction_xy: np.ndarray) -> float:
        """Rise over run of the plane along a horizontal unit direction."""
        return float(-(self.normal[:2] @ direction_xy) / self.normal[2])

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(abs(self.normal[2]), 0, 1))))


def fit_plane(points: np.ndarray, threshold: float, iterations: int, seed: int = 0, weights: np.ndarray | None = None) -> Plane | None:
    """RANSAC plane fit followed by a least-squares refit on the inliers; normal points up.

    With `weights` (for example the reconstruction's per-point confidence), the best plane is the one whose inliers
    carry the most weight, and the refit counts each point by its weight, so doubtful points pull the plane less."""
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    w = np.ones(len(points)) if weights is None else np.clip(np.asarray(weights, float), 0, None)
    best, best_score = None, -1.0
    for _ in range(iterations):
        p0, p1, p2 = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        inliers = np.abs((points - p0) @ normal) < threshold
        score = float(w[inliers].sum())
        if score > best_score:
            best, best_score = inliers, score
    if best is None or best.sum() < 3:
        return None

    inlier_points, inlier_w = points[best], w[best]
    if inlier_w.sum() <= 0:
        inlier_w = np.ones(len(inlier_points))
    centroid = np.average(inlier_points, axis=0, weights=inlier_w)
    normal = np.linalg.svd((inlier_points - centroid) * np.sqrt(inlier_w)[:, None], full_matrices=False)[2][-1]
    normal = normal if normal[2] >= 0 else -normal
    residuals = (inlier_points - centroid) @ normal
    return Plane(normal, float(-normal @ centroid), float(np.sqrt(np.average(residuals ** 2, weights=inlier_w))), float(best.mean()), int(best.sum()))


def principal_direction(xy: np.ndarray) -> np.ndarray:
    """Unit direction of greatest spread of 2D points (e.g. the run of a curb line)."""
    centered = xy - xy.mean(axis=0)
    return np.linalg.svd(centered, full_matrices=False)[2][0]


def ramp_slopes(plane: Plane, curb_direction: np.ndarray) -> tuple[float, float]:
    """(running, cross) slopes as fractions: running is perpendicular to the curb, cross is along it."""
    curb = curb_direction / np.linalg.norm(curb_direction)
    travel = np.array([-curb[1], curb[0]])
    return abs(plane.slope_along(travel)), abs(plane.slope_along(curb))
