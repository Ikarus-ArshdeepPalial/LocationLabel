from math import asin, atan2, cos, degrees, radians, sin, sqrt

EARTH_RADIUS_M = 6_371_000.0


def bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle initial bearing from point 1 to point 2, in degrees clockwise from north."""
    phi1, phi2 = radians(lat1), radians(lat2)
    d_lng = radians(lng2 - lng1)
    x = sin(d_lng) * cos(phi2)
    y = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(d_lng)
    return degrees(atan2(x, y)) % 360


def destination(lat: float, lng: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    delta = distance_m / EARTH_RADIUS_M
    theta = radians(bearing_deg)
    phi1, lambda1 = radians(lat), radians(lng)
    phi2 = asin(sin(phi1) * cos(delta) + cos(phi1) * sin(delta) * cos(theta))
    lambda2 = lambda1 + atan2(sin(theta) * sin(delta) * cos(phi1), cos(delta) - sin(phi1) * sin(phi2))
    return degrees(phi2), (degrees(lambda2) + 540) % 360 - 180


def distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1, phi2 = radians(lat1), radians(lat2)
    a = sin((phi2 - phi1) / 2) ** 2 + cos(phi1) * cos(phi2) * sin(radians(lng2 - lng1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(a))
