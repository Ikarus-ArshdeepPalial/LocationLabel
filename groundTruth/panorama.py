"""Finds the nearest Street View panorama to a ground-truth point and checks a sign faces the camera."""

import requests

from cood_to_sview.geo import bearing, distance
from cood_to_sview.provider import GoogleStreetViewProvider, Pano

from .config import Settings


def find_nearest_pano(provider: GoogleStreetViewProvider, lat: float, lng: float, settings: Settings) -> tuple[Pano, float] | None:
    pano = provider.find_pano(lat, lng, settings.pano_search_radius_m)
    if pano is None:
        return None
    return pano, distance(pano.lat, pano.lng, lat, lng)


def angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def sign_faces_camera(sign_lat: float, sign_lng: float, pano_lat: float, pano_lng: float, facing_direction_deg: float | None, tolerance_deg: float) -> bool:
    """True if the camera is roughly in front of the sign face (facing_direction is the way the sign points)."""
    if facing_direction_deg is None:
        return True  # no facing data recorded, so the filter can't apply
    bearing_to_camera = bearing(sign_lat, sign_lng, pano_lat, pano_lng)
    return angular_difference(bearing_to_camera, facing_direction_deg) <= tolerance_deg
