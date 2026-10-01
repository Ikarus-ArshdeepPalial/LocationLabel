from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt, tan

import numpy as np

from label_objs.config import CLASSES

from .config import Settings
from .geometry import Camera
from .planes import fit_plane, principal_direction, ramp_slopes

GROUND_IDS = [CLASSES.index(c) for c in ("road", "sidewalk", "curb", "curb_ramp", "crosswalk", "pedestrian_area", "terrain")]
ROAD_IDS = [CLASSES.index(c) for c in ("road", "crosswalk")]
SIDEWALK_IDS = [CLASSES.index(c) for c in ("sidewalk", "pedestrian_area", "terrain")]
CURB_ID = CLASSES.index("curb")


@dataclass
class Estimate:
    method: str
    value: float | None
    uncertainty: float = 0.0
    reason: str | None = None
    image: str = ""
    distance_m: float | None = None
    quality: float = 1.0
    depth_scale: float | None = None
    metadata_factor: float = 1.0

    @property
    def applicable(self) -> bool:
        return self.value is not None


def not_applicable(method: str, reason: str) -> Estimate:
    return Estimate(method, None, reason=reason)


def rss(values: Sequence[float]) -> float:
    return sqrt(sum(v * v for v in values))


def sensitivity(fn: Callable[..., float], nominal_args: tuple, deltas: Sequence[tuple]) -> float:
    """Root-sum-square change of fn when each input is perturbed by one standard deviation, one at a time."""
    nominal = fn(*nominal_args)
    return rss([fn(*(a + d for a, d in zip(nominal_args, delta))) - nominal for delta in deltas])


def mad(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def pixel_cloud(camera: Camera, depth: np.ndarray, mask: np.ndarray, stride: int = 2) -> np.ndarray:
    v, u = np.nonzero(mask[::stride, ::stride])
    u, v = u * stride, v * stride
    return camera.points_from_depth(u, v, depth[v, u])


def near_line(points: np.ndarray, line_xy: np.ndarray, radius: float) -> np.ndarray:
    gaps = np.linalg.norm(points[:, None, :2] - line_xy[None, :, :], axis=-1).min(axis=1)
    return points[gaps < radius]


# ---- stop sign ----

def pole_base_problem(semantic: np.ndarray, base: Sequence[int]) -> str | None:
    """Why the pole base can't be trusted as a ground contact point, or None if it can."""
    x, y = base
    if y >= semantic.shape[0] - 2:
        return "pole base cut off by the image edge"
    window = semantic[y + 1:y + 5, max(x - 3, 0):x + 4]
    if np.isin(window, GROUND_IDS).mean() < 0.5:
        return "pole base occluded"
    return None


def sign_flat_ground(camera: Camera, base: Sequence[float], bottom: Sequence[float], s: Settings) -> Estimate:
    def height(h: float, base_v: float, bottom_v: float) -> float:
        cam = camera.with_height(h)
        return h + cam.ground_distance(base[0], base_v)[0] * tan(cam.elevation(bottom[0], bottom_v)[0])

    args = (camera.height, float(base[1]), float(bottom[1]))
    value = height(*args)
    if not np.isfinite(value):
        return not_applicable("flat_ground", "pole base at or above the horizon")
    deltas = [(s.camera_height_sigma_m, 0, 0), (0, s.pixel_sigma, 0), (0, 0, s.pixel_sigma)]
    distance = float(camera.ground_distance(base[0], base[1])[0])
    return Estimate("flat_ground", value, sensitivity(height, args, deltas), distance_m=distance)


def sign_distance(camera: Camera, sign_box: Sequence[float], sign_width_m: float) -> float:
    """Horizontal distance to a roughly camera-facing sign of known width."""
    x1, y1, x2, y2 = sign_box
    depth = camera.fx * sign_width_m / (x2 - x1 + 1)
    center = camera.points_from_depth((x1 + x2) / 2, (y1 + y2) / 2, depth)[0]
    return float(np.hypot(center[0], center[1]))


def sign_from_distance(camera: Camera, distance: float, bottom: Sequence[float], method: str, s: Settings) -> Estimate:
    def height(h: float, bottom_v: float) -> float:
        return h + distance * tan(camera.elevation(bottom[0], bottom_v)[0])

    args = (camera.height, float(bottom[1]))
    deltas = [(s.camera_height_sigma_m, 0), (0, s.pixel_sigma)]
    return Estimate(method, height(*args), sensitivity(height, args, deltas), distance_m=distance)


def sign_reference_scale(
    camera: Camera, sign_box: Sequence[float], base: Sequence[float], bottom: Sequence[float], s: Settings
) -> Estimate:
    x1, y1, x2, y2 = sign_box
    deviation = abs(1 - (y2 - y1 + 1) / (x2 - x1 + 1))
    if deviation > s.max_sign_aspect_deviation:
        return not_applicable("reference_scale", f"sign foreshortened (aspect off by {deviation:.0%})")

    # Pole is vertical in the world, so both ends share one horizontal distance; this handles camera pitch.
    def height(width_px: float, base_v: float, bottom_v: float) -> float:
        d = sign_distance(camera, (x1, y1, x1 + width_px - 1, y2), s.stop_sign_width_m)
        return d * (tan(camera.elevation(bottom[0], bottom_v)[0]) - tan(camera.elevation(base[0], base_v)[0]))

    args = (float(x2 - x1 + 1), float(base[1]), float(bottom[1]))
    deltas = [(s.pixel_sigma, 0, 0), (0, s.pixel_sigma, 0), (0, 0, s.pixel_sigma)]
    value = height(*args)
    uncertainty = rss([sensitivity(height, args, deltas), value * deviation])
    return Estimate("reference_scale", value, uncertainty, distance_m=sign_distance(camera, sign_box, s.stop_sign_width_m))


def sign_point_cloud(
    camera: Camera, depth: np.ndarray, sign_mask: np.ndarray, pole_mask: np.ndarray, base: Sequence[int], bottom: Sequence[float]
) -> Estimate:
    ys, xs = np.nonzero(pole_mask)
    near_base = ys >= base[1] - 5
    sign_depths, base_depths = depth[sign_mask], depth[ys[near_base], xs[near_base]]
    if not len(sign_depths) or not len(base_depths):
        return not_applicable("point_cloud", "no depth on sign or pole base")

    def height(sign_depth: float, base_depth: float) -> float:
        top = camera.points_from_depth(bottom[0], bottom[1], sign_depth)[0]
        foot = camera.points_from_depth(base[0], base[1], base_depth)[0]
        return float(top[2] - foot[2])

    args = (float(np.median(sign_depths)), float(np.median(base_depths)))
    deltas = [(mad(sign_depths), 0), (0, mad(base_depths))]
    foot = camera.points_from_depth(base[0], base[1], args[1])[0]
    return Estimate("point_cloud", height(*args), sensitivity(height, args, deltas), distance_m=float(np.hypot(foot[0], foot[1])))


# ---- curb ----

def curb_profile(semantic: np.ndarray, u: int, v: int, max_face_px: int) -> tuple[float, float] | None:
    """Rows of the curb face's bottom (road side) and top (sidewalk side) edges in column u."""
    column = semantic[:, u]
    row = v
    while row > 0 and column[row] in ROAD_IDS:
        row -= 1
    if column[row] != CURB_ID:
        return None
    bottom = row + 0.5
    while row > 0 and column[row] == CURB_ID:
        row -= 1
    if column[row] not in SIDEWALK_IDS or bottom - row > max_face_px:
        return None
    return bottom, row + 0.5


def curb_flat_ground(camera: Camera, semantic: np.ndarray, curb_line: list[list[int]], s: Settings) -> Estimate:
    heights, distances = [], []
    for u, v in curb_line:
        profile = curb_profile(semantic, u, v, s.max_curb_face_px)
        if not profile:
            continue
        bottom, top = profile
        d = camera.ground_distance(u, bottom)[0]
        if np.isfinite(d):
            heights.append(camera.height + d * tan(camera.elevation(u, top)[0]))
            distances.append(d)
    if len(heights) < 3:
        return not_applicable("flat_ground", "curb top edge not visible")

    values = np.array(heights)
    value = float(np.median(values))
    uncertainty = rss([mad(values), value * s.camera_height_sigma_m / camera.height])
    return Estimate("flat_ground", value, uncertainty, distance_m=float(np.median(distances)))


def curb_point_cloud(camera: Camera, depth: np.ndarray, semantic: np.ndarray, curb_line: list[list[int]], s: Settings) -> Estimate:
    line = np.array(curb_line)
    line_xy = camera.points_from_depth(line[:, 0], line[:, 1], depth[line[:, 1], line[:, 0]])[:, :2]
    road = near_line(pixel_cloud(camera, depth, np.isin(semantic, ROAD_IDS)), line_xy, s.curb_neighborhood_m)
    side = near_line(pixel_cloud(camera, depth, np.isin(semantic, SIDEWALK_IDS)), line_xy, s.curb_neighborhood_m)
    if len(road) < 30 or len(side) < 30:
        return not_applicable("point_cloud", "too few road or sidewalk points near the curb")

    road_plane = fit_plane(road, s.ransac_threshold_m, s.ransac_iterations)
    side_plane = fit_plane(side, s.ransac_threshold_m, s.ransac_iterations)
    if road_plane is None or side_plane is None:
        return not_applicable("point_cloud", "plane fit failed")
    offsets = side_plane.z_at(line_xy) - road_plane.z_at(line_xy)
    value = float(np.median(offsets))
    uncertainty = rss([road_plane.rms, side_plane.rms, mad(offsets)])
    return Estimate("point_cloud", value, uncertainty, distance_m=float(np.median(np.linalg.norm(line_xy, axis=1))))


# ---- curb ramp ----

def ramp_point_cloud(
    camera: Camera, depth: np.ndarray, ramp_mask: np.ndarray, curb_line: list[list[int]], s: Settings
) -> tuple[Estimate, Estimate]:
    """(running, cross) slope estimates in percent from a plane fitted to the ramp's points."""
    if len(curb_line) < 2:
        reason = "no curb line next to the ramp"
        return not_applicable("point_cloud", reason), not_applicable("point_cloud", reason)

    line = np.array(curb_line)
    line_xy = camera.points_from_depth(line[:, 0], line[:, 1], depth[line[:, 1], line[:, 0]])[:, :2]
    points = near_line(pixel_cloud(camera, depth, ramp_mask, stride=1), line_xy, s.curb_neighborhood_m)
    plane = fit_plane(points, s.ransac_threshold_m, s.ransac_iterations) if len(points) >= 50 else None
    if plane is None:
        reason = "too few ramp points for a plane fit"
        return not_applicable("point_cloud", reason), not_applicable("point_cloud", reason)

    running, cross = ramp_slopes(plane, principal_direction(line_xy))
    extent = float(np.ptp(points[:, :2] @ principal_direction(points[:, :2])))
    uncertainty = 100 * plane.rms / max(extent / 2, 0.25)
    distance = float(np.median(np.linalg.norm(points[:, :2], axis=1)))
    return (
        Estimate("point_cloud", 100 * running, uncertainty, distance_m=distance),
        Estimate("point_cloud", 100 * cross, uncertainty, distance_m=distance),
    )
