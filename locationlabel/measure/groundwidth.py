"""Widths of flat surfaces (a sidewalk, a driveway) seen from street level, measured two ways that check each other.

Ground plane: every pixel of the surface is projected onto the road plane from the camera height and angles, the same
geometry as the flat-ground height estimate. Depth: the pixels are lifted with the depth model. A width is the
spread of the surface across the curb's direction, taken in 0.5 m slabs along the curb so that one odd place (an
apron, a gap) does not decide it.

The ground-plane method falls apart for rays close to the horizon (a one degree error in pitch moves a point at 2 degrees
below the horizon by more than 100 m), so pixels that are too close to it, or too far away, are left out.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import Camera
from .planes import principal_direction

MIN_DEPRESSION_DEG = 3.0   # rays flatter than this below the horizon are not used
SLAB_M = 0.5
AGREE = 0.25               # the two methods agree when they are within this fraction of each other


@dataclass
class StripWidth:
    width_m: float
    line: list[list[float]]  # pixel endpoints of the measured cross-section
    slabs: int


def project(camera: Camera, depth: np.ndarray | None, xs: np.ndarray, ys: np.ndarray, max_range_m: float, method: str):
    """(ground xy, pixel xs, pixel ys) of the usable pixels for one method, or None if the method cannot be used."""
    if method == "depth":
        if depth is None:
            return None
        points = camera.points_from_depth(xs, ys, depth[ys, xs])
        ok = np.isfinite(points).all(axis=1)
    else:
        points = camera.ground_points(xs, ys)
        ok = np.isfinite(points).all(axis=1) & (-np.degrees(camera.elevation(xs, ys)) >= MIN_DEPRESSION_DEG)
    xy = points[:, :2]
    with np.errstate(invalid="ignore"):
        ok &= np.hypot(*(xy - camera.center[:2]).T) <= max_range_m
    return xy[ok], xs[ok], ys[ok]


def strip_width(xy: np.ndarray, px: np.ndarray, py: np.ndarray, boundary_xy: np.ndarray) -> StripWidth | None:
    if len(boundary_xy) < 20 or len(xy) < 300:
        return None  # no curb edge in range: there is no direction to measure across
    along = principal_direction(boundary_xy)
    across = np.array([-along[1], along[0]])
    centered = xy - xy.mean(axis=0)
    a_pos, c_pos = centered @ along, centered @ across
    slabs = np.floor(a_pos / SLAB_M).astype(int)
    found = []
    for slab in np.unique(slabs):
        members = np.nonzero(slabs == slab)[0]
        if len(members) >= 40:
            lo, hi = np.percentile(c_pos[members], [2, 98])
            found.append((hi - lo, members, lo, hi))
    if len(found) < 3:
        return None
    found.sort(key=lambda f: f[0])
    width, members, lo, hi = found[len(found) // 2]
    ends = [int(members[np.argmin(np.abs(c_pos[members] - edge))]) for edge in (lo, hi)]
    return StripWidth(float(width), [[float(px[i]), float(py[i])] for i in ends], len(found))


def two_method_width(camera: Camera, depth: np.ndarray | None, xs, ys, boundary_xs, boundary_ys, max_range_m: float) -> dict:
    """{"ground_plane": StripWidth, "depth": StripWidth} for whichever methods could measure the surface."""
    result = {}
    for name, method in (("ground_plane", "ground"), ("depth", "depth")):
        surface, edge = project(camera, depth, xs, ys, max_range_m, method), project(camera, depth, boundary_xs, boundary_ys, max_range_m, method)
        if surface and edge:
            strip = strip_width(surface[0], surface[1], surface[2], edge[0])
            if strip:
                result[name] = strip
    return result


def combine(strips: dict, shape: tuple[int, int]) -> dict | None:
    """One width from the methods: their mean when they agree (that is what makes it certain), otherwise the
    ground-plane value on its own, marked uncertain. 'cut_off' means the cross-section ends at the edge of the photo."""
    if not strips:
        return None
    ground, depth = strips.get("ground_plane"), strips.get("depth")
    chosen = ground or depth
    agree = bool(ground and depth and abs(ground.width_m - depth.width_m) <= AGREE * max(ground.width_m, depth.width_m))
    width = (ground.width_m + depth.width_m) / 2 if agree else chosen.width_m
    height, image_width = shape
    cut_off = any(x <= 2 or y <= 2 or x >= image_width - 3 or y >= height - 3 for x, y in chosen.line)
    return {
        "width_m": round(float(width), 2),
        "methods": {name: round(strip.width_m, 2) for name, strip in strips.items()},
        "agree": agree,
        "cut_off": bool(cut_off),
        "certain": bool(agree and not cut_off),
        "line": chosen.line,
    }
