import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import radians, tan
from pathlib import Path
from typing import Any

import requests

from common.cache import JsonCache
from common.progress import progress
from common.record import load_record, record_path, update_record

from .config import Settings, ViewSpec
from .contact_sheet import make_contact_sheet
from .geo import bearing, destination, distance
from .provider import Pano, StreetViewError, StreetViewProvider
from .target import LatLng, Target, google_rooftop, resolve_target

logger = logging.getLogger(__name__)


@dataclass
class View:
    pano: Pano
    role: str
    spec: ViewSpec
    heading: float

    @property
    def pitch(self) -> float:
        return self.spec.pitch

    @property
    def filename(self) -> str:
        short_id = re.sub(r"\W", "", self.pano.pano_id)[:10]
        return f"pano{short_id}_h{self.heading:.0f}_p{self.pitch:.0f}.jpg"


class StreetViewStage:
    def __init__(
        self,
        settings: Settings,
        provider: StreetViewProvider,
        cache: JsonCache,
        session: requests.Session,
        dry_run: bool = False,
        force: bool = False,
        refine_with_google: bool = False,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._cache = cache
        self._session = session
        self._dry_run = dry_run
        self._force = force
        self._refine = refine_with_google

    def run(self, address: str) -> str:
        path = record_path(self._settings.output_dir, address)
        if not path.exists():
            logger.error("No Stage 1 record for %r at %s, run the geocode stage first", address, path)
            return "error"

        try:
            section = self._process(load_record(path), path.parent)
        except (requests.RequestException, StreetViewError) as exc:
            section = {"status": "error", "message": str(exc)}
        except Exception as exc:
            logger.exception("Street View stage failed for %r", address)
            section = {"status": "error", "message": f"Unexpected error: {exc}"}

        if not self._dry_run:
            update_record(path, {"streetview": section})
        log = logger.info if section["status"] in ("ok", "dry_run") else logger.warning
        log("%r: streetview %s - %s", address, section["status"], section.get("message", ""))
        return section["status"]

    def _process(self, record: dict[str, Any], address_dir: Path) -> dict[str, Any]:
        if record.get("status") != "ok" or record.get("lat") is None:
            return {"status": "error", "message": "Stage 1 has no coordinates for this address"}

        target = resolve_target(record, self._settings.side_offset_m, self._refined_point(record))
        found = self._find_main_pano(target.adjusted)
        if not found:
            radii = ", ".join(map(str, self._settings.search_radii))
            return {"status": "no_imagery", "message": f"No outdoor panorama within {radii} m", "target": asdict(target)}

        main, radius = found
        neighbors = self._find_neighbors(main, target)
        views = self._plan_views(main, neighbors, target.adjusted)
        images_dir = address_dir / "images"

        if self._dry_run:
            pending = [v for v in views if self._force or not (images_dir / v.filename).exists()]
            message = f"{1 + len(neighbors)} panos, {len(views)} views, {len(pending)} image requests would be made"
            return {"status": "dry_run", "message": message, "image_requests_planned": len(pending)}

        requests_before = self._provider.image_requests
        images, failures = self._fetch(views, images_dir, target.adjusted)
        if images:
            labels = [(images_dir / i["filename"], f"{i['role']} {i['pano_id'][:8]} h{i['heading']:.0f} p{i['pitch']:.0f}")
                      for i in images]
            make_contact_sheet(labels, address_dir / "contact_sheet.jpg")

        status = "ok" if not failures else "partial" if images else "error"
        return {
            "status": status,
            "message": f"{len(images)}/{len(views)} views saved",
            "provider": self._provider.name,
            "target": asdict(target),
            "panos": [self._pano_entry(main, "main", target.adjusted, radius)]
                     + [self._pano_entry(p, "neighbor", target.adjusted, self._settings.neighbor_step_m) for p in neighbors],
            "images": images,
            "failed_views": failures,
            "contact_sheet": "contact_sheet.jpg" if images else None,
            "image_requests": self._provider.image_requests - requests_before,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _refined_point(self, record: dict[str, Any]) -> LatLng | None:
        if not self._refine or not self._settings.google_api_key:
            return None
        return google_rooftop(self._session, self._settings.google_api_key, record["formatted_address"], self._settings.timeout)

    def _find_pano(self, lat: float, lng: float, radius: float) -> Pano | None:
        key = f"{self._provider.name}:{lat:.6f},{lng:.6f}:{radius}"
        cached = None if self._force else self._cache.get(key)
        if cached:
            return Pano(**cached)
        pano = self._provider.find_pano(lat, lng, radius)
        if pano:
            self._cache.put(key, asdict(pano))
        return pano

    def _find_main_pano(self, point: LatLng) -> tuple[Pano, int] | None:
        for radius in self._settings.search_radii:
            pano = self._find_pano(point.lat, point.lng, radius)
            if pano:
                return pano, radius
            logger.info("No panorama within %d m of (%s, %s)", radius, point.lat, point.lng)
        return None

    def _find_neighbors(self, main: Pano, target: Target) -> list[Pano]:
        # Without street geometry, assume the street runs perpendicular to the line of sight to the house.
        along_street = target.street_bearing
        if along_street is None:
            along_street = bearing(main.lat, main.lng, target.adjusted.lat, target.adjusted.lng) + 90

        neighbors: list[Pano] = []
        for direction in (along_street, along_street + 180):
            lat, lng = destination(main.lat, main.lng, direction, self._settings.neighbor_step_m)
            pano = self._find_pano(lat, lng, self._settings.neighbor_step_m)
            if pano and pano.pano_id not in {main.pano_id, *(n.pano_id for n in neighbors)}:
                neighbors.append(pano)
        return neighbors

    def _plan_views(self, main: Pano, neighbors: list[Pano], point: LatLng) -> list[View]:
        plan = [(main, "main", self._settings.main_views)] + [(n, "neighbor", self._settings.neighbor_views) for n in neighbors]
        views = []
        for pano, role, specs in plan:
            to_target = bearing(pano.lat, pano.lng, point.lat, point.lng)
            views += [View(pano, role, spec, round((to_target + spec.heading_offset) % 360, 1)) for spec in specs]
        return views

    def _fetch(self, views: list[View], images_dir: Path, point: LatLng) -> tuple[list[dict], list[dict]]:
        images_dir.mkdir(parents=True, exist_ok=True)
        s = self._settings
        images, failures = [], []
        bar = progress(views, "  fetching")
        for view in bar:
            bar.set_postfix_str(f"{self._provider.image_requests} requests")
            path = images_dir / view.filename
            if self._force or not path.exists():
                try:
                    path.write_bytes(self._provider.fetch_image(view.pano.pano_id, view.heading, view.pitch, s.fov, s.image_width, s.image_height))
                except (requests.RequestException, StreetViewError) as exc:
                    logger.warning("View %s of pano %s failed: %s", view.spec.name, view.pano.pano_id, exc)
                    failures.append({"view": view.spec.name, "pano_id": view.pano.pano_id, "heading": view.heading, "pitch": view.pitch, "error": str(exc)})
                    continue
            images.append(self._image_entry(view, point))
        return images, failures

    def _image_entry(self, view: View, point: LatLng) -> dict[str, Any]:
        s = self._settings
        focal_px = s.image_width / (2 * tan(radians(s.fov) / 2))
        return {
            "filename": view.filename,
            "view": view.spec.name,
            "role": view.role,
            "pano_id": view.pano.pano_id,
            "pano_lat": view.pano.lat,
            "pano_lng": view.pano.lng,
            "capture_date": view.pano.capture_date,
            "heading": view.heading,
            "pitch": view.pitch,
            "fov": s.fov,
            "width": s.image_width,
            "height": s.image_height,
            "intrinsics": {"fx": focal_px, "fy": focal_px, "cx": s.image_width / 2, "cy": s.image_height / 2},
            "distance_to_target_m": round(distance(view.pano.lat, view.pano.lng, point.lat, point.lng), 2),
        }

    @staticmethod
    def _pano_entry(pano: Pano, role: str, point: LatLng, radius: float) -> dict[str, Any]:
        return {
            **asdict(pano),
            "role": role,
            "search_radius_m": radius,
            "distance_to_target_m": round(distance(pano.lat, pano.lng, point.lat, point.lng), 2),
        }
