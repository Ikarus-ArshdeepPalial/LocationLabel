from typing import Any

from cood_to_sview.geo import distance


def representative_point(geometry: dict[str, Any]) -> tuple[float, float] | None:
    """(lat, lng) for a point/polyline/polygon geometry: the point itself, or a line's length-weighted midpoint."""
    if geometry is None:
        return None
    if "x" in geometry and "y" in geometry:
        return geometry["y"], geometry["x"]

    rings_or_paths = geometry.get("paths") or geometry.get("rings")
    if not rings_or_paths:
        return None
    return polyline_midpoint([(lat, lng) for lng, lat in rings_or_paths[0]])


def polyline_midpoint(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    if len(points) == 1:
        return points[0]

    segment_lengths = [distance(*points[i], *points[i + 1]) for i in range(len(points) - 1)]
    total = sum(segment_lengths)
    if total == 0:
        return points[0]

    target, covered = total / 2, 0.0
    for (lat1, lng1), (lat2, lng2), length in zip(points, points[1:], segment_lengths):
        if covered + length >= target:
            fraction = (target - covered) / length if length else 0
            return lat1 + (lat2 - lat1) * fraction, lng1 + (lng2 - lng1) * fraction
        covered += length
    return points[-1]
