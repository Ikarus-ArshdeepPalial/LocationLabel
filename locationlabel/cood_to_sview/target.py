import logging
from dataclasses import dataclass
from typing import Any

import requests

from .geo import bearing, destination, distance

GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

logger = logging.getLogger(__name__)


@dataclass
class LatLng:
    lat: float
    lng: float


@dataclass
class Target:
    original: LatLng
    adjusted: LatLng
    method: str
    street_bearing: float | None


def street_bearing(segment: dict[str, Any] | None) -> float | None:
    if not segment:
        return None
    start, end = segment["from"], segment["to"]
    if distance(start["lat"], start["lng"], end["lat"], end["lng"]) < 1:
        return None
    return bearing(start["lat"], start["lng"], end["lat"], end["lng"])


def side_offset(point: LatLng, segment: dict[str, Any], street_dir: float, offset_m: float) -> LatLng:
    toward_house = street_dir - 90 if segment["side"] == "L" else street_dir + 90
    return LatLng(*destination(point.lat, point.lng, toward_house, offset_m))


def google_rooftop(session: requests.Session, api_key: str, address: str, timeout: float) -> LatLng | None:
    try:
        response = session.get(GOOGLE_GEOCODE_URL, params={"address": address, "key": api_key}, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.warning("Google refinement failed for %r: %s", address, exc)
        return None

    if data.get("status") != "OK":
        return None
    top = data["results"][0]
    is_us = any("country" in c["types"] and c["short_name"] == "US" for c in top["address_components"])
    if not is_us or top["geometry"]["location_type"] != "ROOFTOP":
        return None
    return LatLng(top["geometry"]["location"]["lat"], top["geometry"]["location"]["lng"])


def resolve_target(
    record: dict[str, Any],
    offset_m: float,
    refine: LatLng | None = None,
) -> Target:
    original = LatLng(record["lat"], record["lng"])
    segment = record.get("street_segment")
    street_dir = street_bearing(segment)

    if record["provider"] == "google" and record["match_quality"] == "ROOFTOP":
        return Target(original, original, "stage1_rooftop", street_dir)
    if record["provider"] == "ground_truth":
        # Ground-truth rows already give the exact object location; never offset it.
        return Target(original, original, "ground_truth_exact", street_dir)
    if refine:
        return Target(original, refine, "google_rooftop", street_dir)
    if segment and street_dir is not None and segment.get("side") in ("L", "R"):
        return Target(original, side_offset(original, segment, street_dir, offset_m), "census_side_offset", street_dir)
    return Target(original, original, "none", street_dir)
