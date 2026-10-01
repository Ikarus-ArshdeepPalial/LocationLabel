import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import atan2, cos, degrees, radians, sin
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common.progress import progress
from common.record import load_record, record_path, update_record

from label_objs.config import CLASSES

from .config import METADATA_PENALTY, METHOD_DEPENDENCIES, Settings
from .depth import DepthCache, DepthModel, depth_scale
from .estimators import (
    ROAD_IDS,
    Estimate,
    curb_flat_ground,
    curb_point_cloud,
    curb_profile,
    not_applicable,
    pole_base_problem,
    ramp_point_cloud,
    sign_distance,
    sign_flat_ground,
    sign_from_distance,
    sign_point_cloud,
    sign_reference_scale,
)
from .fusion import ramp_block, scalar_block
from .geometry import Camera, direction, latlng_to_local, triangulate
from .limits import Check, evaluate, load_limits
from .multiview import RampView, VggtReconstructor, measure_ramp, pick_ramp
from .planes import principal_direction
from .outputs import GREY, PURPLE, RED, Drawing, draw_measurements, mask_outline, ramp_profile, write_ply

logger = logging.getLogger(__name__)


@dataclass
class View:
    name: str
    camera: Camera
    pano_id: str
    pano_xy: np.ndarray | None
    metadata_sources: dict[str, str]
    rgb: np.ndarray
    semantic: np.ndarray
    instances: list[dict]
    address_dir: Path
    depth: np.ndarray | None = None
    depth_scale: float | None = None
    depth_problem: str | None = None

    def mask(self, instance: dict) -> np.ndarray:
        return cv2.imread(str(self.address_dir / instance["mask"]), cv2.IMREAD_GRAYSCALE) > 0

    def instance(self, instance_id: str) -> dict:
        return next(i for i in self.instances if i["instance_id"] == instance_id)

    @property
    def located(self) -> bool:
        """Camera position and heading are known, so image directions can be placed on the map."""
        return self.pano_xy is not None and self.metadata_sources.get("heading_deg") != "default"

    def metadata_factor(self, method: str) -> float:
        dependencies = METHOD_DEPENDENCIES.get(method, {})
        return float(np.prod([METADATA_PENALTY.get(self.metadata_sources.get(f), 1.0) ** w for f, w in dependencies.items()]))


@dataclass
class Observation:
    view: View
    instance: dict
    estimates: list[Estimate]
    position: np.ndarray | None
    azimuth: float
    outline: list[list[float]]
    cross: list[Estimate] = field(default_factory=list)


class MeasurementStage:
    def __init__(
        self,
        settings: Settings,
        depth_model: DepthModel | None,
        force: bool = False,
        export_pointcloud: bool = False,
        reconstructor: VggtReconstructor | None = None,
    ) -> None:
        self._s = settings
        self._reconstructor = reconstructor
        self._limits = load_limits(settings.limits_file)
        self._checks: list[Check] = []
        self._depth_model = depth_model
        self._force = force
        self._export_pointcloud = export_pointcloud

    def run(self, address: str) -> dict[str, Any]:
        path = record_path(self._s.output_dir, address)
        if not path.exists():
            logger.error("No record for %r at %s, run the earlier stages first", address, path)
            return {"status": "error", "message": "no record"}

        try:
            section = self._process(load_record(path), path.parent)
        except Exception as exc:
            logger.exception("Measurement stage failed for %r", address)
            section = {"status": "error", "message": f"Unexpected error: {exc}"}

        update_record(path, {"measurements": section})
        log = logger.info if section["status"] == "ok" else logger.warning
        log("%r: measurements %s - %s", address, section["status"], section.get("message", ""))
        return section

    def _process(self, record: dict[str, Any], address_dir: Path) -> dict[str, Any]:
        per_image = (record.get("detections") or {}).get("per_image")
        if not per_image:
            return {"status": "error", "message": "No Stage 3 detections, run the detect stage first"}

        metas = {m["filename"]: m for m in record["streetview"]["images"]}
        origin = self._origin(record)
        depth_cache = DepthCache(self._depth_model, address_dir / "depth", self._force) if self._depth_model else None
        signs, ramps, curbs, poles, sidewalks, images, failed = [], [], [], [], [], [], []
        curb_detected = False
        self._checks = []
        ramp_views: list[RampView] = []
        # Photos in the `ramp` folder measure the ramp together; ordinary images then skip the single-photo slope.
        has_ramp_group = any(m.get("group") == "ramp" for m in metas.values())

        for entry in progress(per_image, "  measuring"):
            try:
                view = self._load_view(entry, metas[entry["image"]], address_dir, origin)
                if metas[entry["image"]].get("group") == "ramp":
                    instance = pick_ramp(view.instances)
                    if instance:
                        ramp_views.append(RampView(view.name, view.rgb, view.mask(instance), view.semantic == CLASSES.index("road"), instance, instance["confidence"]))
                    images.append({"image": view.name, "group": "ramp", "ramp_found": instance is not None})
                    continue
                classes = {i["class"] for i in view.instances}
                if classes & {"stop_sign", "curb", "curb_ramp", "pole", "sidewalk"}:
                    self._attach_depth(view, depth_cache)
                signs += [self._observe_sign(view, i) for i in view.instances if i["class"] == "stop_sign"]
                if not has_ramp_group:
                    ramps += [self._observe_ramp(view, i) for i in view.instances if i["class"] == "curb_ramp"]
                curbs += [self._observe_curb(view, i) for i in view.instances if i["class"] == "curb" and len(i.get("curb_line") or []) >= 3]
                sign_poles = {i["pole"]["instance_id"] for i in view.instances if i["class"] == "stop_sign" and i.get("pole")}
                poles += [p for i in view.instances if i["class"] == "pole" and i["instance_id"] not in sign_poles and (p := self._observe_pole(view, i))]
                biggest = max((i for i in view.instances if i["class"] == "sidewalk"), key=lambda i: i["area_px"], default=None)
                if biggest and (w := self._observe_sidewalk(view, biggest)):
                    sidewalks.append(w)
                curb_detected |= "curb" in classes
                images.append({
                    "image": view.name,
                    "depth_scale": round(view.depth_scale, 3) if view.depth_scale else None,
                    "depth_scale_flag": view.depth_scale is not None and abs(view.depth_scale - 1) > self._s.depth_scale_warn,
                    "depth_problem": view.depth_problem,
                })
                if self._export_pointcloud and view.depth is not None:
                    write_ply(address_dir / "pointclouds" / f"{Path(view.name).stem}.ply", view.camera, view.depth, view.semantic, view.rgb)
            except Exception as exc:
                logger.exception("Measuring %s failed", entry["image"])
                failed.append({"image": entry["image"], "error": str(exc)})

        sign_groups, ramp_groups = self._group(signs), self._group(ramps)
        positions = [self._triangulate(group) for group in sign_groups]
        sign_objects = [
            with_total({**scalar_block("stop_sign_height", True, [e for o in g for e in o.estimates], self._s), **self._object_info(g, p)}, self._s.stop_sign_width_m)
            for g, p in zip(sign_groups, positions)
        ]
        ramp_objects = [
            {**ramp_block(True, [e for o in g for e in o.estimates], [e for o in g for e in o.cross], self._s), **self._object_info(g, None)}
            for g in ramp_groups
        ]
        stop_sign = self._address_level(sign_objects, "stop_sign_height", lambda o: o["confidence"])
        ramp = self._address_level(ramp_objects, "ramp", lambda o: o["running_slope_pct"])
        group_drawings: list[tuple[RampView, Drawing]] = []
        if has_ramp_group:
            ramp, group_drawings = self._measure_ramp_group(ramp_views, metas)
        curb = scalar_block("curb_height", curb_detected, [e for o in curbs for e in o.estimates], self._s)

        self._draw(address_dir, sign_groups, sign_objects, ramp_groups, ramp_objects, curbs, curb, poles, group_drawings, sidewalks)
        status = "error" if not images else "partial" if failed else "ok"
        return {
            "status": status,
            "message": f"{len(images)} images measured, {len(failed)} failed",
            "config_used": self._config_used(),
            "stop_sign_height": stop_sign,
            "curb_height": curb,
            "ramp": ramp,
            "poles": [info for _, info in poles],
            "sidewalks": [info for _, info in sidewalks],
            "images": images,
            "failed_images": failed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **({"limits": self._limits_section()} if self._limits else {}),
        }

    def _limits_section(self) -> dict[str, Any]:
        return {
            "rules": [asdict(rule) for rule in self._limits],
            "checks": [c.to_dict() for c in self._checks],
            "exceeded": [c.to_dict() for c in self._checks if c.exceeded],
        }

    def _check(self, measure: str, value_base: float | None, subject: str, certain: bool) -> list[Check]:
        """Checks a value against the regional limits and remembers the result for the summary."""
        if value_base is None or not self._limits:
            return []
        checks = evaluate(self._limits, measure, value_base, subject, certain)
        self._checks += checks
        return checks

    def _slope_checks(self, block: dict, kind: str, subject: str) -> list[Check]:
        certain = block["source"] == "measured"
        running = block["running_slope_pct"] if certain else block.get("candidate_running_slope_pct")
        cross = block["cross_slope_pct"] if certain else block.get("candidate_cross_slope_pct")
        checks = self._check(f"{kind}_running_slope", running, subject, certain)
        if kind == "ramp":  # a driveway's cross slope is too small to read from a photo, so only its slope is checked
            checks += self._check("ramp_cross_slope", cross, subject, certain)
        return checks

    def _width_checks(self, block: dict, subject: str) -> list[Check]:
        """Driveway width against its limits. A width cut off by the frame can only be longer, so it cannot show a driveway too narrow."""
        width = block.get("width_m")
        if not width:
            return []
        certain = block["source"] == "measured"
        return self._check("driveway_max_width", width, subject, certain) + self._check("driveway_min_width", width, subject, certain and not block.get("width_cut_off"))

    def _measure_ramp_group(self, views: list[RampView], metas: dict[str, dict]) -> tuple[dict[str, Any], list[tuple[RampView, Drawing]]]:
        """One ramp measured from all the photos in the `ramp` folder."""
        if self._reconstructor is None:
            return ramp_block(False, [], [], self._s), []
        heights = [metas[v.name].get("camera_height_m") or self._s.camera_height_m for v in views]
        measurement = measure_ramp(views, self._reconstructor, float(np.mean(heights)) if heights else self._s.camera_height_m, self._s)
        block = {
            **ramp_block(True, measurement.running, measurement.cross, self._s),
            **measurement.detail,
            "images": [v.name for v in measurement.views],
            "instance_ids": [v.instance["instance_id"] for v in measurement.views],
        }
        kind = "driveway" if any(v.instance.get("kind") == "driveway" for v in measurement.views) else "ramp"
        block["kind"] = kind
        checks = self._slope_checks(block, kind, "ramp photos") + (self._width_checks(block, "ramp photos") if kind == "driveway" else [])
        label = limit_label(ramp_label(block, builtin=not self._limits), checks, block)
        drawings = []
        if label:
            label = (label[0], label[1] + width_text(block), label[2])
            for view in measurement.views:
                profile = ramp_profile(view.ramp, view.instance.get("curb_line") or [])
                drawings.append((view, Drawing(profile, *label, view.instance.get("outline") or mask_outline(view.ramp))))
        return self._address_level([block], "ramp", lambda o: o["running_slope_pct"]), drawings

    def _origin(self, record: dict[str, Any]) -> tuple[float, float] | None:
        target = (record["streetview"].get("target") or {}).get("adjusted")
        if target:
            return target["lat"], target["lng"]
        located = [m for m in record["streetview"]["images"] if m.get("pano_lat") is not None]
        return (located[0]["pano_lat"], located[0]["pano_lng"]) if located else None

    def _load_view(self, entry: dict, meta: dict, address_dir: Path, origin: tuple[float, float] | None) -> View:
        bgr = cv2.imread(str(address_dir / "images" / entry["image"]))
        semantic = cv2.imread(str(address_dir / entry["semantic"]), cv2.IMREAD_UNCHANGED)
        if bgr is None or semantic is None:
            raise FileNotFoundError(f"missing image or semantic map for {entry['image']}")
        return View(
            name=entry["image"],
            camera=Camera.from_metadata(meta, meta.get("camera_height_m") or self._s.camera_height_m),
            pano_id=meta["pano_id"],
            pano_xy=latlng_to_local(meta["pano_lat"], meta["pano_lng"], *origin) if origin and meta.get("pano_lat") is not None else None,
            metadata_sources=meta.get("metadata_sources", {}),
            rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            semantic=semantic,
            instances=entry["instances"],
            address_dir=address_dir,
        )

    def _attach_depth(self, view: View, cache: DepthCache | None) -> None:
        if cache is None:
            view.depth_problem = "depth model disabled"
            return
        try:
            depth = cache.get(Path(view.name).stem, view.rgb)
        except Exception as exc:
            logger.warning("Depth failed for %s: %s", view.name, exc)
            view.depth_problem = f"depth model failed: {exc}"
            return
        scale = depth_scale(depth, view.camera, np.isin(view.semantic, ROAD_IDS), self._s.depth_ground_max_distance_m)
        if scale is None:
            view.depth_problem = "no nearby road to calibrate depth scale"
        view.depth, view.depth_scale = depth * (scale or 1.0), scale

    def _tag(self, estimates: list[Estimate], view: View, instance: dict) -> None:
        for e in estimates:
            e.image, e.quality = view.name, instance["confidence"]
            e.metadata_factor = view.metadata_factor(e.method)
            if e.method == "point_cloud":
                e.depth_scale = view.depth_scale

    def _observe_sign(self, view: View, inst: dict) -> Observation:
        cam, bottom = view.camera, inst["bottom_center"]
        sign_mask = view.mask(inst)
        ys, xs = np.nonzero(sign_mask)
        sign_box = (xs.min(), ys.min(), xs.max(), ys.max())
        base = inst["pole"]["base_pixel"] if inst.get("pole") else None
        problem = "no pole associated with the sign" if base is None else pole_base_problem(view.semantic, base)

        if problem:
            estimates = [not_applicable(m, problem) for m in ("flat_ground", "reference_scale", "point_cloud")]
        else:
            estimates = [sign_flat_ground(cam, base, bottom, self._s), sign_reference_scale(cam, sign_box, base, bottom, self._s)]
            if view.depth is not None:
                pole_mask = view.mask(view.instance(inst["pole"]["instance_id"]))
                estimates.append(sign_point_cloud(cam, view.depth, sign_mask, pole_mask, base, bottom))
            else:
                estimates.append(not_applicable("point_cloud", view.depth_problem))
        self._tag(estimates, view, inst)

        azimuth = float(cam.azimuth(bottom[0], bottom[1])[0])
        ground = cam.ground_points(*base)[0] if not problem else np.full(3, np.nan)
        local = ground[:2] if np.isfinite(ground).all() else sign_distance(cam, sign_box, self._s.stop_sign_width_m) * direction(azimuth)
        position = view.pano_xy + local if view.located else None
        # Measurement line from the pole base to the sign bottom, drawn just beside the pole.
        offset = (sign_box[2] - sign_box[0]) / 2
        top = sign_box[1]
        line = [[base[0] + offset, base[1]], [bottom[0] + offset, top]] if base else [[bottom[0] + offset, top]]
        return Observation(view, inst, estimates, position, azimuth, line)

    def _observe_pole(self, view: View, inst: dict) -> tuple[Observation, dict] | None:
        """Pole height from its lowest to its highest mask pixel; a top at the image edge makes it a lower bound."""
        mask = view.mask(inst)
        ys, xs = np.nonzero(mask)
        base, top = [int(xs[ys.argmax()]), int(ys.max())], [int(xs[ys.argmin()]), int(ys.min())]
        if pole_base_problem(view.semantic, base) or base[1] - top[1] < 0.15 * mask.shape[0]:
            return None
        estimates = [sign_flat_ground(view.camera, base, top, self._s)]
        if view.depth is not None:
            top_mask = mask & (np.arange(mask.shape[0])[:, None] <= top[1] + 5)
            estimates.append(sign_point_cloud(view.camera, view.depth, top_mask, mask, base, top))
        values = [e.value for e in estimates if e.value is not None and e.value > 0]
        if not values:
            return None
        height, agree = float(np.mean(values)), len(values) == 2 and abs(values[0] - values[1]) <= 0.2 * max(values)
        cut_off = top[1] <= 2
        info = {"image": view.name, "height_m": round(height, 2), "methods": {e.method: round(e.value, 2) for e in estimates if e.value}, "top_cut_off": bool(cut_off), "agree": bool(agree)}
        offset = 0.02 * mask.shape[1]
        line = [[base[0] + offset, base[1]], [top[0] + offset, top[1]]]
        sign = self._sign_on_pole(view, mask, base, top, height)
        if sign:
            info["sign"] = sign["info"]
            line = [[base[0] + offset, base[1]], [sign["top_point"][0] + offset, sign["top_point"][1]]]
            info["sign_line"] = [[sign["box"][2] + offset / 2, sign["box"][3]], [sign["box"][2] + offset / 2, sign["box"][1]]]
        return Observation(view, inst, estimates, None, 0.0, line), info

    def _sign_on_pole(self, view: View, pole_mask: np.ndarray, base: list[int], top: list[int], pole_height: float) -> dict | None:
        """A stop sign sitting on top of this pole: the pole's mask ends where the sign begins, so the pole alone
        measures only up to the bottom of the sign. Returns the sign's own height and the total height."""
        for inst in view.instances:
            if inst["class"] != "stop_sign":
                continue
            ys, xs = np.nonzero(view.mask(inst))
            if not len(ys):
                continue
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            reach = box[3] - box[1]
            if not (box[0] - reach <= top[0] <= box[2] + reach and box[1] < top[1] and top[1] - reach * 0.5 <= box[3] <= top[1] + reach):
                continue
            top_point = [(box[0] + box[2]) // 2, box[1]]
            estimates = [sign_flat_ground(view.camera, base, top_point, self._s)]
            if view.depth is not None:
                estimates.append(sign_point_cloud(view.camera, view.depth, view.mask(inst), pole_mask, base, top_point))
            totals = [e.value for e in estimates if e.value is not None and e.value > 0]
            face = self._s.stop_sign_width_m
            measured = float(np.mean(totals)) - pole_height if totals else None
            use_measured = measured is not None and 0.5 * face <= measured <= 1.5 * face
            sign_h = measured if use_measured else face
            return {
                "box": box,
                "top_point": top_point,
                "info": {
                    "sign_height_m": round(sign_h, 3),
                    "sign_height_source": "measured" if use_measured else "standard",
                    "bottom_height_m": round(pole_height, 3),
                    "total_height_m": round(pole_height + sign_h, 3),
                },
            }
        return None

    def _observe_sidewalk(self, view: View, inst: dict) -> tuple[Observation, dict] | None:
        """Sidewalk width: across the curb's direction, in slabs along the curb; the middle slab width is reported, so a
        driveway apron or a gap in one place does not decide it. Only points within a short range of the camera count."""
        if view.depth is None:
            return None
        mask = view.mask(inst)
        gap = max(9, round(0.015 * max(mask.shape)))  # the curb lies between the road and the sidewalk
        edge = cv2.dilate(np.isin(view.semantic, [CLASSES.index("road"), CLASSES.index("curb")]).astype(np.uint8), np.ones((2 * gap + 1,) * 2, np.uint8)) > 0
        inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

        def cloud(selection: np.ndarray):
            ys, xs = np.nonzero(selection)
            step = max(1, len(ys) // 20000)
            ys, xs = ys[::step], xs[::step]
            if not len(ys):
                return np.empty((0, 2)), xs, ys
            points = view.camera.points_from_depth(xs, ys, view.depth[ys, xs])
            near = np.isfinite(points).all(axis=1) & (np.hypot(*(points[:, :2] - view.camera.center[:2]).T) <= self._s.sidewalk_max_range_m)
            return points[near, :2], xs[near], ys[near]

        boundary, _, _ = cloud(mask & edge)
        xy, xs, ys = cloud(inner)
        if len(boundary) < 100 or len(xy) < 300:
            return None  # no curb edge in view: there is no direction to measure across
        along = principal_direction(boundary)
        across = np.array([-along[1], along[0]])
        a_pos, c_pos = (xy - xy.mean(axis=0)) @ along, (xy - xy.mean(axis=0)) @ across
        slabs = np.floor(a_pos / 0.5).astype(int)
        widths = []
        for slab in np.unique(slabs):
            members = np.nonzero(slabs == slab)[0]
            if len(members) >= 40:
                lo, hi = np.percentile(c_pos[members], [2, 98])
                widths.append((hi - lo, members, lo, hi))
        if len(widths) < 3:
            return None
        widths.sort(key=lambda w: w[0])
        width, members, lo, hi = widths[len(widths) // 2]
        ends = [int(members[np.argmin(np.abs(c_pos[members] - edge_value))]) for edge_value in (lo, hi)]
        line = [[float(xs[i]), float(ys[i])] for i in ends]
        height, image_width = mask.shape
        cut_off = any(x <= 2 or y <= 2 or x >= image_width - 3 or y >= height - 3 for x, y in line)
        reliable = view.depth_scale is not None and abs(view.depth_scale - 1) <= self._s.depth_scale_warn
        info = {"image": view.name, "width_m": round(float(width), 2), "cut_off": bool(cut_off), "certain": bool(reliable and not cut_off), "slabs": len(widths)}
        return Observation(view, inst, [], None, 0.0, line), info

    def _observe_ramp(self, view: View, inst: dict) -> Observation:
        cam, mask = view.camera, view.mask(inst)
        if inst.get("mask_source") == "point":
            reason = "ramp detected, but no ramp region was segmented to measure"
            running, cross = not_applicable("point_cloud", reason), not_applicable("point_cloud", reason)
            self._tag([running, cross], view, inst)
            return Observation(view, inst, [running], None, float(cam.azimuth(*inst["point"])[0]), [inst["point"]], [cross])
        if view.depth is None:
            running, cross = not_applicable("point_cloud", view.depth_problem), not_applicable("point_cloud", view.depth_problem)
        else:
            running, cross = ramp_point_cloud(cam, view.depth, mask, inst.get("curb_line") or nearest_curb_line(view, inst), self._s)
        self._tag([running, cross], view, inst)
        ground = cam.ground_points(*inst["centroid"])[0]
        position = view.pano_xy + ground[:2] if view.located and np.isfinite(ground).all() else None
        profile = ramp_profile(mask, inst.get("curb_line") or [])
        inst = {**inst, "outline": inst.get("outline") or mask_outline(mask)}  # a ramp is drawn as an outline, like a driveway
        return Observation(view, inst, [running], position, float(cam.azimuth(*inst["centroid"])[0]), profile, [cross])

    def _observe_curb(self, view: View, inst: dict) -> Observation:
        line = inst["curb_line"]
        estimates = [curb_flat_ground(view.camera, view.semantic, line, self._s)]
        if view.depth is not None:
            estimates.append(curb_point_cloud(view.camera, view.depth, view.semantic, line, self._s))
        else:
            estimates.append(not_applicable("point_cloud", view.depth_problem))
        self._tag(estimates, view, inst)
        return Observation(view, inst, estimates, None, 0.0, self._curb_tick(view.semantic, line))

    def _curb_tick(self, semantic: np.ndarray, line: list[list[int]]) -> list[list[float]]:
        """Vertical tick on the curb face (road edge to curb top) in the middle of the visible curb."""
        faces = [(u, face) for u, v in line if (face := curb_profile(semantic, u, v, self._s.max_curb_face_px))]
        if not faces:
            u, v = line[len(line) // 2]
            return [[u, v]]
        u, (bottom, top) = faces[len(faces) // 2]
        middle, half = (bottom + top) / 2, max((bottom - top) / 2, 6)
        return [[u, middle + half], [u, middle - half]]

    def _group(self, observations: list[Observation]) -> list[list[Observation]]:
        """Greedy clustering by estimated ground position; observations without a position join the first group."""
        groups: list[list[Observation]] = []
        for obs in observations:
            # Two objects in one photo are never the same object, so each photo joins a group at most once.
            open_groups = [g for g in groups if all(o.view.name != obs.view.name for o in g)]
            if obs.position is None and open_groups:
                open_groups[0].append(obs)
                continue
            for group in open_groups:
                centers = [o.position for o in group if o.position is not None]
                if obs.position is not None and centers and np.linalg.norm(np.mean(centers, axis=0) - obs.position) < self._s.group_radius_m:
                    group.append(obs)
                    break
            else:
                groups.append([obs])
        return groups

    def _triangulate(self, group: list[Observation]) -> np.ndarray | None:
        by_pano: dict[str, list[Observation]] = defaultdict(list)
        for obs in group:
            if obs.view.located:
                by_pano[obs.view.pano_id].append(obs)

        point = None
        if len(by_pano) >= 2:
            origins = np.array([obs[0].view.pano_xy for obs in by_pano.values()])
            azimuths = [circular_mean([o.azimuth for o in obs]) for obs in by_pano.values()]
            point = triangulate(origins, azimuths, self._s.min_triangulation_angle_deg)

        for obs in group:
            if point is None:
                if not obs.view.located:
                    reason = "camera position or heading unknown"
                elif len(by_pano) < 2:
                    reason = "sign seen from only one panorama"
                else:
                    reason = "rays nearly parallel or diverging"
                estimate = not_applicable("triangulation", reason)
            else:
                distance = float(np.linalg.norm(point - obs.view.pano_xy))
                estimate = sign_from_distance(obs.view.camera, distance, obs.instance["bottom_center"], "triangulation", self._s)
            self._tag([estimate], obs.view, obs.instance)
            obs.estimates.append(estimate)
        return point

    @staticmethod
    def _object_info(group: list[Observation], triangulated: np.ndarray | None) -> dict[str, Any]:
        positions = [o.position for o in group if o.position is not None]
        position = triangulated if triangulated is not None else (np.mean(positions, axis=0) if positions else None)
        return {
            "images": sorted({o.view.name for o in group}),
            "instance_ids": [o.instance["instance_id"] for o in group],
            "position_m": [round(float(c), 2) for c in position] if position is not None else None,
            "triangulated": triangulated is not None,
        }

    def _address_level(self, objects: list[dict], kind: str, rank) -> dict[str, Any]:
        """Signs: the most confident measured sign. Ramps: the steepest measured ramp (the compliance worst case)."""
        if not objects:
            return ramp_block(False, [], [], self._s) if kind == "ramp" else scalar_block(kind, False, [], self._s)
        measured = [o for o in objects if o["source"] == "measured"]
        chosen = max(measured or objects, key=rank if measured else lambda o: o["confidence"])
        return {**chosen, "objects": objects}

    def _draw(self, address_dir: Path, sign_groups, sign_objects, ramp_groups, ramp_objects, curbs, curb, poles, group_drawings=(), sidewalks=()) -> None:
        drawings: dict[str, tuple[Any, list[Drawing]]] = {}
        for view, drawing in group_drawings:
            drawings.setdefault(view.name, (view, []))[1].append(drawing)

        def add(obs: Observation, label: tuple[str, str, tuple[int, int, int]] | None) -> None:
            if label is None:
                return  # a default value is not a measurement, so nothing is drawn
            drawings.setdefault(obs.view.name, (obs.view, []))[1].append(Drawing(obs.outline, *label, obs.instance.get("outline")))

        face = self._s.stop_sign_width_m
        for group, block in zip(sign_groups, sign_objects):
            bottom = block["value_m"] if block["source"] == "measured" else block.get("candidate_value_m")
            certain = block["source"] == "measured"
            subject = ", ".join(block["images"])
            checks = self._check("stop_sign_bottom_height", bottom, subject, certain) + self._check("stop_sign_total_height", None if bottom is None else bottom + face, subject, certain)
            for obs in group:
                add(obs, limit_label(sign_label(block, face), checks, block))
        for group, block in zip(ramp_groups, ramp_objects):
            kind = "driveway" if any(o.instance.get("kind") == "driveway" for o in group) else "ramp"
            checks = self._slope_checks(block, kind, ", ".join(block["images"]))
            for obs in group:
                label = ramp_label(block, kind == "driveway", builtin=not self._limits)
                add(obs, limit_label((label[0], f"driveway · {label[1]}", label[2]) if kind == "driveway" and label else label, checks, block))
        longest_curb: dict[str, Observation] = {}
        for obs in curbs:
            current = longest_curb.get(obs.view.name)
            if current is None or len(obs.instance["curb_line"]) > len(current.instance["curb_line"]):
                longest_curb[obs.view.name] = obs
        curb_value = curb["value_m"] if curb["source"] == "measured" else curb.get("candidate_value_m")
        curb_checks = self._check("curb_height", curb_value, "curb", curb["source"] == "measured")
        for obs in longest_curb.values():
            add(obs, limit_label(height_label(curb), curb_checks, curb))

        for obs, info in poles:
            sure = info["agree"] and not info["top_cut_off"]
            checks = self._check("pole_height", info["height_m"], info["image"], sure)
            sign, sign_line = info.get("sign"), info.pop("sign_line", None)
            if sign is None:
                add(obs, limit_label(pole_label(info), checks))
                continue
            checks += self._check("stop_sign_bottom_height", sign["bottom_height_m"], info["image"], sure)
            checks += self._check("stop_sign_total_height", sign["total_height_m"], info["image"], sure)
            value, detail, color = pole_label(info)
            total = f"{round(sign['total_height_m'] * 100)} cm"
            add(obs, limit_label((total if sure else f"~{total}", f"to sign bottom {round(sign['bottom_height_m'] * 100)} cm · {detail}", color), checks))
            standard = sign["sign_height_source"] == "standard"
            sign_drawing = Drawing(sign_line, f"{'~' if standard else ''}{round(sign['sign_height_m'] * 100)} cm", "sign only · standard size" if standard else "sign only", PURPLE if sure else GREY)
            drawings.setdefault(obs.view.name, (obs.view, []))[1].append(sign_drawing)

        for obs, info in sidewalks:
            checks = self._check("sidewalk_width", info["width_m"], info["image"], info["certain"])
            value = f"{'≥ ' if info['cut_off'] else ''}{round(info['width_m'] * 100)} cm"
            add(obs, limit_label((value, "sidewalk width", PURPLE if info["certain"] else GREY), checks))

        (address_dir / "measured").mkdir(exist_ok=True)
        rendered = {name: draw_measurements(view.rgb, items) for name, (view, items) in drawings.items()}
        for name, image in rendered.items():
            cv2.imwrite(str(address_dir / "measured" / f"{Path(name).stem}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if rendered:
            best = max(rendered, key=lambda name: len(drawings[name][1]))
            cv2.imwrite(str(address_dir / "measured.jpg"), cv2.cvtColor(rendered[best], cv2.COLOR_RGB2BGR))

    def _config_used(self) -> dict[str, Any]:
        s = self._s
        return {
            "camera_height_m": s.camera_height_m,
            "camera_height_sigma_m": s.camera_height_sigma_m,
            "stop_sign_width_m": s.stop_sign_width_m,
            "depth_model": self._depth_model.version if self._depth_model else None,
            "confidence_threshold": s.confidence_threshold,
            "mad_k": s.mad_k,
            "min_triangulation_angle_deg": s.min_triangulation_angle_deg,
            "curb_neighborhood_m": s.curb_neighborhood_m,
            "limits": asdict(s.limits),
            "plausible": asdict(s.plausible),
        }


def circular_mean(azimuths_deg: list[float]) -> float:
    return degrees(atan2(sum(sin(radians(a)) for a in azimuths_deg), sum(cos(radians(a)) for a in azimuths_deg))) % 360


def nearest_curb_line(view: View, ramp: dict) -> list[list[int]]:
    """Curb line of the curb closest to the ramp, for ramps whose own edge doesn't touch the road mask."""
    lines = [i["curb_line"] for i in view.instances if i["class"] == "curb" and len(i.get("curb_line") or []) >= 2]
    center = np.array(ramp["centroid"])
    return min(lines, key=lambda line: np.linalg.norm(np.array(line) - center, axis=1).min(), default=[])


def pole_label(info: dict) -> tuple[str, str, tuple[int, int, int]]:
    """Grey and approximate unless flat-ground and depth agree and the whole pole is in view."""
    sure = info["agree"] and not info["top_cut_off"]
    value = f"{'' if sure else '≥ ' if info['top_cut_off'] else '~'}{round(info['height_m'] * 100)} cm"
    if len(info["methods"]) < 2:
        note = "1 method"
    else:
        note = "2 methods agree" if info["agree"] else "average of 2 methods"  # height_m is already their mean
    return value, f"pole · {note}", PURPLE if sure else GREY


def width_text(block: dict) -> str:
    """' · width 3.1 m' for a ramp or driveway whose width was measured; '≥' when the photos cut it off."""
    if not block.get("width_m"):
        return ""
    cut = block.get("width_cut_off")
    return f" · width {'≥ ' if cut else ''}{block['width_m']:.1f} m / {block['width_m'] / 0.3048:.1f} ft" + (" (photo edge cuts it)" if cut else "")


def confidence_text(block: dict) -> str:
    return f"confidence {block['confidence']:.2f}/1"


def with_total(block: dict, face_m: float) -> dict:
    """A sign block plus its total height: the measured height to the bottom of the sign, plus the sign face."""
    bottom = block["value_m"] if block["source"] == "measured" else block.get("candidate_value_m")
    return {**block, "sign_face_m": face_m, "total_height_m": None if bottom is None else round(bottom + face_m, 4)}


def sign_label(block: dict, face_m: float) -> tuple[str, str, tuple[int, int, int]] | None:
    """The sign's total height (pole and sign), with the height to the bottom of the sign beside it."""
    if block["source"] == "measured":
        bottom = block["value_m"]
        uncertainty = max(1, round(block["uncertainty_m"] * 100))
        return f"{round((bottom + face_m) * 100)} cm", f"to sign bottom {round(bottom * 100)} cm · ±{uncertainty} cm · {confidence_text(block)}", PURPLE
    if block.get("candidate_value_m") is not None:
        bottom = block["candidate_value_m"]
        return f"~{round((bottom + face_m) * 100)} cm", f"to sign bottom ~{round(bottom * 100)} cm · {confidence_text(block)}", GREY
    return None


def limit_label(label, checks: list[Check], block: dict | None = None):
    """When a value is over its limit by more than the margin, the label shows that value and the limit in red."""
    over = next((c for c in checks if c.exceeded), None)
    if over is None or label is None:
        return label
    tail = f" · {confidence_text(block)}" if block and "confidence" in block else ""
    if over.rule.measure.endswith("_width"):  # the angle stays the headline; the width problem goes in the description
        return label[0], f"width {over.value_text} · {over.message}{tail}", RED
    return over.value_text, over.message + tail, RED


def height_label(block: dict) -> tuple[str, str, tuple[int, int, int]] | None:
    """(value, detail, colour) for a stop sign or curb height label; low-confidence estimates are grey with a ~."""
    if block["source"] == "measured":
        return f"{round(block['value_m'] * 100)} cm", f"±{max(1, round(block['uncertainty_m'] * 100))} cm · {confidence_text(block)}", PURPLE
    if block.get("candidate_value_m") is not None:
        return f"~{round(block['candidate_value_m'] * 100)} cm", confidence_text(block), GREY
    return None


# Driveway aprons are never this flat, so a reading under 1 degree is shown scaled up (chosen by the user, not measured).
DRIVEWAY_FLAT_SCALES = ((0.5, 11.0), (1.0, 6.0))


def driveway_degrees(degrees: float) -> float:
    for below, scale in DRIVEWAY_FLAT_SCALES:
        if degrees < below:
            return degrees * scale
    return degrees


def ramp_label(block: dict, driveway: bool = False, builtin: bool = True) -> tuple[str, str, tuple[int, int, int]] | None:
    label = _ramp_label(block, builtin)
    if label is None:
        return None
    raw = float(np.degrees(np.arctan((block.get("candidate_running_slope_pct") or block["running_slope_pct"]) / 100)))
    if not driveway or block["source"] != "measured" and block.get("candidate_running_slope_pct") is None or driveway_degrees(raw) == raw:
        return label
    value, detail, color = label
    return f"{'~' if value.startswith('~') else ''}{driveway_degrees(raw):.1f}°", detail, color


def _ramp_label(block: dict, builtin: bool = True) -> tuple[str, str, tuple[int, int, int]] | None:
    if block.get("candidate_running_slope_pct") is not None:
        estimate = np.degrees(np.arctan(block["candidate_running_slope_pct"] / 100))
        return f"~{estimate:.1f}°", confidence_text(block), GREY
    if block["source"] != "measured":
        return None
    detail = f"{block['running_slope_pct']:.1f}% run · {block['cross_slope_pct']:.1f}% cross · {confidence_text(block)}"
    if builtin and block["exceeds_limit"]:
        detail += " · over ADA"
    confident = builtin and any(e["exceeds"] and e["beyond_uncertainty"] for e in block["exceedance"].values())  # your own limits replace the built-in ADA flag
    return f"{block['running_slope_deg']:.1f}°", detail, RED if confident else PURPLE
