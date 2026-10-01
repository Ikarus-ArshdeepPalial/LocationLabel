"""Pinhole camera in a local ENU world frame (x east, y north, z up) with the ground at z = 0."""

from dataclasses import dataclass, replace
from functools import cached_property
from math import cos, radians, sin
from typing import Any

import numpy as np

from cood_to_sview.geo import bearing, distance


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    heading: float
    pitch: float
    height: float
    width: int = 640
    image_height: int = 640

    @classmethod
    def from_metadata(cls, meta: dict[str, Any], camera_height: float) -> "Camera":
        k = meta["intrinsics"]
        return cls(k["fx"], k["fy"], k["cx"], k["cy"], meta["heading"], meta["pitch"], camera_height, meta["width"], meta["height"])

    def with_height(self, height: float) -> "Camera":
        return replace(self, height=height)

    @property
    def center(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.height])

    @cached_property
    def rotation(self) -> np.ndarray:
        """Camera (x right, y down, z forward) to world. Street View pitch is positive upward, no roll."""
        h, p = radians(self.heading), radians(self.pitch)
        forward = np.array([sin(h) * cos(p), cos(h) * cos(p), sin(p)])
        right = np.array([cos(h), -sin(h), 0.0])
        up = np.cross(right, forward)
        return np.column_stack([right, -up, forward])

    def rays_camera(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rays with camera-frame z = 1, so multiplying by z-depth gives the camera-frame point."""
        u, v = np.atleast_1d(u).astype(float), np.atleast_1d(v).astype(float)
        return np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=-1)

    def rays_world(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.rays_camera(u, v) @ self.rotation.T

    def elevation(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        rays = self.rays_world(u, v)
        return np.arctan2(rays[:, 2], np.hypot(rays[:, 0], rays[:, 1]))

    def azimuth(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compass bearing of each pixel's ray, degrees clockwise from north."""
        rays = self.rays_world(u, v)
        return np.degrees(np.arctan2(rays[:, 0], rays[:, 1])) % 360

    def ground_depth(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Camera z-depth where each ray meets flat ground; NaN for rays at or above the horizon."""
        down = self.rays_world(u, v)[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(down < 0, -self.height / down, np.nan)

    def ground_points(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.points_from_depth(u, v, self.ground_depth(u, v))

    def ground_distance(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        points = self.ground_points(u, v)
        return np.hypot(points[:, 0], points[:, 1])

    def points_from_depth(self, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
        return self.center + (self.rays_camera(u, v) * np.atleast_1d(depth)[:, None]) @ self.rotation.T


def latlng_to_local(lat: float, lng: float, origin_lat: float, origin_lng: float) -> np.ndarray:
    """East/north offset in metres of (lat, lng) from the origin, via great-circle distance and bearing."""
    d = distance(origin_lat, origin_lng, lat, lng)
    if d < 1e-6:
        return np.zeros(2)
    b = radians(bearing(origin_lat, origin_lng, lat, lng))
    return np.array([d * sin(b), d * cos(b)])


def direction(azimuth_deg: float) -> np.ndarray:
    return np.array([sin(radians(azimuth_deg)), cos(radians(azimuth_deg))])


def max_ray_angle(azimuths_deg: list[float]) -> float:
    """Largest angle in degrees between any two ray directions."""
    diffs = [abs((a - b + 180) % 360 - 180) for i, a in enumerate(azimuths_deg) for b in azimuths_deg[i + 1:]]
    return max(diffs, default=0.0)


def triangulate(origins: np.ndarray, azimuths_deg: list[float], min_angle_deg: float) -> np.ndarray | None:
    """Least-squares intersection of 2D bearing rays; None if they are nearly parallel or meet behind a camera."""
    if len(origins) < 2 or max_ray_angle(azimuths_deg) < min_angle_deg:
        return None
    a, b = np.zeros((2, 2)), np.zeros(2)
    for origin, azimuth in zip(origins, azimuths_deg):
        d = direction(azimuth)
        projector = np.eye(2) - np.outer(d, d)
        a += projector
        b += projector @ origin
    point = np.linalg.solve(a, b)
    if any((point - o) @ direction(az) <= 0 for o, az in zip(origins, azimuths_deg)):
        return None
    return point
