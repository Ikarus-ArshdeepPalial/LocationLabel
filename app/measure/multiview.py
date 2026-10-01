"""Ramp slope from several photos of one ramp taken at different positions.

VGGT rebuilds the scene (cameras and depth) from the photos. The ramp's plane is then measured against the road's
plane, so the result needs neither a metric depth scale nor a gravity estimate. Cross-checking the views against
each other gives the confidence.
"""

import logging
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .estimators import Estimate, not_applicable
from .planes import Plane, fit_plane, principal_direction

logger = logging.getLogger(__name__)

THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party" / "vggt"
SIDE = 518  # VGGT's working resolution


@dataclass
class RampView:
    name: str
    rgb: np.ndarray
    ramp: np.ndarray        # boolean mask of the ramp surface
    road: np.ndarray        # boolean mask of the road
    instance: dict
    confidence: float


@dataclass
class Reconstruction:
    points: np.ndarray       # (views, SIDE, SIDE, 3) in VGGT's own scale and frame
    confidence: np.ndarray   # (views, SIDE, SIDE)
    centers: np.ndarray      # (views, 3) camera positions


def use_bundled_vggt() -> None:
    if str(THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY))


class VggtReconstructor:
    def __init__(self, model_id: str, device: str) -> None:
        self._model_id = model_id
        self._device = device

    @cached_property
    def _model(self):
        use_bundled_vggt()
        from vggt.models.vggt import VGGT

        logger.info("Loading %s on %s (VGGT is licensed for non-commercial use unless the Commercial checkpoint is used)", self._model_id, self._device)
        return VGGT.from_pretrained(self._model_id).to(self._device).eval()

    def reconstruct(self, images: list[np.ndarray]) -> Reconstruction:
        """images: uint8 RGB, all SIDE x SIDE."""
        import torch

        use_bundled_vggt()
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div(255).to(self._device)
        dtype = torch.bfloat16 if self._device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.inference_mode(), torch.autocast(device_type=self._device.split(":")[0], dtype=dtype, enabled=self._device.startswith("cuda")):
            out = self._model(batch)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(out["pose_enc"], batch.shape[-2:])
        extrinsic, intrinsic = extrinsic[0].float().cpu().numpy(), intrinsic[0].float().cpu().numpy()
        depth = out["depth"][0].float().cpu().numpy()
        points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
        centers = np.stack([-e[:, :3].T @ e[:, 3] for e in extrinsic])
        return Reconstruction(points, out["depth_conf"][0].float().cpu().numpy(), centers)


def crop_around(view: RampView, factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A square window centred on the ramp and `factor` times its size, resized to VGGT's resolution.

    Cropping puts more pixels on the ramp than the whole photo would (a tested 2x zoom was the most accurate).
    """
    height, width = view.ramp.shape
    ys, xs = np.nonzero(view.ramp)
    side = int(min(max(max(np.ptp(xs), np.ptp(ys)) * factor, 0.3 * min(height, width)), min(height, width)))
    x0 = int(np.clip((xs.min() + xs.max()) / 2 - side / 2, 0, width - side))
    y0 = int(np.clip((ys.min() + ys.max()) / 2 - side / 2, 0, height - side))

    def cut(array: np.ndarray, interpolation: int) -> np.ndarray:
        return cv2.resize(array[y0:y0 + side, x0:x0 + side], (SIDE, SIDE), interpolation=interpolation)

    return cut(view.rgb, cv2.INTER_AREA), cut(view.ramp.astype(np.uint8), cv2.INTER_NEAREST) > 0, cut(view.road.astype(np.uint8), cv2.INTER_NEAREST) > 0


def usable(mask: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Interior pixels of a mask, where the reconstruction is reasonably confident."""
    inside = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return inside & (confidence >= np.percentile(confidence, 30))


@dataclass
class Slopes:
    running_pct: float
    cross_pct: float
    uncertainty_pct: float
    distance_m: float
    ramp_points: int
    width_m: float = 0.0   # extent of the ramp along the curb
    length_m: float = 0.0  # extent of the ramp up from the road


def estimate_slopes(rec: Reconstruction, ramps: list[np.ndarray], roads: list[np.ndarray], views: list[int], camera_height_m: float, s: Settings) -> Slopes | None:
    """Ramp slopes in percent, relative to the road plane, from the chosen views of a reconstruction."""
    ramp_sel = [usable(ramps[i], rec.confidence[i]) for i in views]
    road_sel = [usable(roads[i], rec.confidence[i]) for i in views]
    ramp_pts = np.concatenate([rec.points[i][m] for i, m in zip(views, ramp_sel)])
    road_pts = np.concatenate([rec.points[i][m] for i, m in zip(views, road_sel)])
    if len(ramp_pts) < 40 or len(road_pts) < 100:
        return None
    # The reconstruction's own confidence weights each point, so doubtful points pull the planes less.
    ramp_w = np.concatenate([rec.confidence[i][m] for i, m in zip(views, ramp_sel)])
    road_w = np.concatenate([rec.confidence[i][m] for i, m in zip(views, road_sel)])

    centers = rec.centers[views]
    scale = float(np.median(np.linalg.norm(ramp_pts - centers.mean(axis=0), axis=1)))
    rng = np.random.default_rng(0)

    def take(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(points) <= 20000:
            return points, weights
        pick = rng.choice(len(points), 20000, replace=False)
        return points[pick], weights[pick]

    sample_road, sample_road_w = take(road_pts, road_w)
    road = fit_plane(sample_road, 0.004 * scale, s.ransac_iterations, weights=sample_road_w)
    if road is None:
        return None

    origin = -road.offset * road.normal
    normal = road.normal if (centers.mean(axis=0) - origin) @ road.normal > 0 else -road.normal
    height_units = float(np.mean((centers - origin) @ normal))
    if height_units <= 0:
        return None

    boundary_pts = np.concatenate([rec.points[i][usable(roads[i] & (cv2.dilate(ramps[i].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0), rec.confidence[i])] for i in views])
    along_source = boundary_pts if len(boundary_pts) >= 20 else ramp_pts
    along = principal_direction(along_source)
    along = along - (along @ normal) * normal
    along /= np.linalg.norm(along)
    frame = np.stack([along, np.cross(normal, along), normal])

    sample_ramp, sample_ramp_w = take(ramp_pts, ramp_w)
    local = (sample_ramp - origin) @ frame.T
    plane: Plane | None = fit_plane(local, 0.004 * scale, s.ransac_iterations, weights=sample_ramp_w)
    if plane is None:
        return None
    running, cross = abs(plane.slope_along(np.array([0.0, 1.0]))), abs(plane.slope_along(np.array([1.0, 0.0])))
    depth_extent = max(float(np.ptp(local[:, 1])), 1e-9)
    metres_per_unit = camera_height_m / height_units
    distance = float(np.median(np.linalg.norm(ramp_pts[:, None, :] - centers[None, :, :], axis=2))) * metres_per_unit
    width_units, length_units = (float(np.subtract(*np.percentile(local[:, axis], [98, 2]))) for axis in (0, 1))
    return Slopes(100 * running, 100 * cross, 100 * plane.rms / (depth_extent / 2), distance, int(plane.n_points), width_units * metres_per_unit, length_units * metres_per_unit)


def ramp_surface_confidence(rec: Reconstruction, ramps: list[np.ndarray]) -> float:
    """How sure the reconstruction is about the ramp itself, compared with the rest of the photos, from 0.3 to 1.

    1 when the ramp is reconstructed at least as confidently as the scene around it; lower when the ramp is the part
    it is unsure about (a plain, textureless or shadowed surface), so the slope's confidence drops with it."""
    on_ramp = np.concatenate([rec.confidence[i][ramps[i]] for i in range(len(ramps))])
    if not len(on_ramp):
        return 0.3
    ratio = float(np.median(on_ramp) / max(float(np.median(rec.confidence)), 1e-9))
    return float(np.clip(ratio, 0.3, 1.0))


@dataclass
class RampMeasurement:
    running: list[Estimate]
    cross: list[Estimate]
    views: list[RampView]
    detail: dict


def pick_ramp(instances: list[dict]) -> dict | None:
    """The instance to measure in a view: a segmented ramp region beats a driveway, which beats a bare point."""
    rank = {"segmentation": 2, "driveway": 1}
    candidates = [i for i in instances if i["class"] == "curb_ramp" and rank.get(i.get("mask_source"), 0) > 0]
    return max(candidates, key=lambda i: (rank[i["mask_source"]], i["area_px"]), default=None)


def measure_ramp(views: list[RampView], reconstructor: VggtReconstructor, camera_height_m: float, s: Settings) -> RampMeasurement:
    """Running and cross slope estimates from all views, plus leave-one-out estimates that show how much they agree."""
    if len(views) < 2:
        reason = "a multi-view ramp measurement needs at least 2 photos with a detected ramp"
        return RampMeasurement([not_applicable("multiview", reason)], [not_applicable("multiview", reason)], views, {})

    views = views[: s.multiview_max_views]
    crops = [crop_around(v, s.multiview_crop_factor) for v in views]
    try:
        rec = reconstructor.reconstruct([c[0] for c in crops])
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(views) <= 2:
            raise
        logger.warning("GPU memory ran out with %d views; retrying with %d", len(views), len(views) - 1)
        return measure_ramp(views[:-1], reconstructor, camera_height_m, s)
    ramps, roads = [c[1] for c in crops], [c[2] for c in crops]

    everything = list(range(len(views)))
    full = estimate_slopes(rec, ramps, roads, everything, camera_height_m, s)
    if full is None:
        reason = "the ramp or the road had too few reliable 3D points"
        return RampMeasurement([not_applicable("multiview", reason)], [not_applicable("multiview", reason)], views, {})

    subsets = [[j for j in everything if j != i] for i in everything] if len(views) >= 3 else [everything]
    running, cross = [], []
    surface = ramp_surface_confidence(rec, ramps)
    quality = float(np.mean([v.confidence for v in views])) * surface
    for subset in subsets:
        result = estimate_slopes(rec, ramps, roads, subset, camera_height_m, s)
        if result is None:
            continue
        for target, value in ((running, result.running_pct), (cross, result.cross_pct)):
            target.append(Estimate("multiview", value, result.uncertainty_pct, image="+".join(views[j].name for j in subset), distance_m=result.distance_m, quality=quality))
    touches_frame = all(v.ramp[0].any() or v.ramp[-1].any() or v.ramp[:, 0].any() or v.ramp[:, -1].any() for v in views)  # cut off only if no photo shows the whole ramp
    detail = {"method": "multiview", "reference": "road plane", "n_views": len(views), "ramp_points": full.ramp_points, "width_m": round(full.width_m, 2), "length_m": round(full.length_m, 2), "width_cut_off": bool(touches_frame), "surface_confidence": round(surface, 2), "all_views_running_slope_pct": round(full.running_pct, 2), "all_views_cross_slope_pct": round(full.cross_pct, 2)}
    return RampMeasurement(running or [not_applicable("multiview", "no view subset gave a plane")], cross or [not_applicable("multiview", "no view subset gave a plane")], views, detail)
