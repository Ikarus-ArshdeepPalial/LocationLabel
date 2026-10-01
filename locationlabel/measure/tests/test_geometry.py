from math import atan2, degrees, radians, tan

import numpy as np
import pytest

from measure.geometry import Camera, latlng_to_local, max_ray_angle, triangulate
from measure.planes import fit_plane, ramp_slopes
from measure.tests.synthetic import project

CAMERA = Camera(fx=320, fy=320, cx=320, cy=320, heading=0, pitch=0, height=2.5)


def test_principal_ray_points_along_heading():
    assert CAMERA.rays_world(320, 320)[0] == pytest.approx([0, 1, 0])
    east = Camera(320, 320, 320, 320, heading=90, pitch=0, height=2.5)
    assert east.rays_world(320, 320)[0] == pytest.approx([1, 0, 0])
    assert east.azimuth(320, 320)[0] == pytest.approx(90)


def test_back_projection_offsets():
    ray = CAMERA.rays_world(320 + 160, 320 - 80)[0]
    assert ray == pytest.approx([0.5, 1, 0.25])  # right and up in the world
    assert degrees(CAMERA.elevation(320, 320 - 320)[0]) == pytest.approx(45)


@pytest.mark.parametrize("heading, pitch", [(0, 0), (37, -20), (250, 12)])
def test_projection_round_trip(heading, pitch):
    camera = Camera(320, 320, 320, 320, heading, pitch, 2.5)
    point = camera.points_from_depth(100, 500, 7.0)[0]
    assert project(camera, point) == pytest.approx((100, 500))


def test_ground_intersection_level_camera():
    v = 320 + 320 * 2.5 / 10  # ray dropping 2.5 m over 10 m
    point = CAMERA.ground_points(320, v)[0]
    assert point == pytest.approx([0, 10, 0])
    assert CAMERA.ground_distance(320, v)[0] == pytest.approx(10)


def test_ground_intersection_pitched_camera():
    camera = Camera(320, 320, 320, 320, heading=0, pitch=-20, height=2.5)
    assert camera.ground_distance(320, 320)[0] == pytest.approx(2.5 / tan(radians(20)))
    assert np.isnan(camera.ground_distance(320, 0)[0])  # above the horizon


def test_latlng_to_local():
    east, north = latlng_to_local(30.0, -97.0 + 0.001, 30.0, -97.0)
    assert east == pytest.approx(96.3, abs=0.2) and north == pytest.approx(0, abs=0.01)


def azimuth_to(origin, point) -> float:
    return degrees(atan2(point[0] - origin[0], point[1] - origin[1])) % 360


def test_triangulation_recovers_point():
    target = np.array([10.0, 20.0])
    origins = np.array([[0.0, 0.0], [15.0, 0.0], [30.0, 5.0]])
    azimuths = [azimuth_to(o, target) for o in origins]

    assert triangulate(origins, azimuths, 10) == pytest.approx(target)


def test_triangulation_rejects_nearly_parallel_rays():
    target = np.array([0.0, 100.0])
    origins = np.array([[0.0, 0.0], [5.0, 0.0]])
    azimuths = [azimuth_to(o, target) for o in origins]

    assert max_ray_angle(azimuths) < 10
    assert triangulate(origins, azimuths, 10) is None


@pytest.mark.parametrize("curb_angle_deg", [0, 35])
def test_plane_fit_and_slope_decomposition(curb_angle_deg):
    rng = np.random.default_rng(1)
    curb = np.array([np.cos(radians(curb_angle_deg)), np.sin(radians(curb_angle_deg))])
    travel = np.array([-curb[1], curb[0]])
    along, across = rng.uniform(-1, 1, (2, 2000))
    xy = np.outer(along, curb) + np.outer(across, travel)
    z = 0.08 * across + 0.02 * along + rng.normal(0, 0.005, 2000)  # 8% running, 2% cross
    points = np.column_stack([xy, z])
    points[:300, 2] += rng.uniform(0.2, 0.5, 300)  # outliers, e.g. a parked car

    plane = fit_plane(points, threshold=0.02, iterations=300)
    running, cross = ramp_slopes(plane, curb)

    assert running == pytest.approx(0.08, abs=0.003)
    assert cross == pytest.approx(0.02, abs=0.003)
    assert plane.inlier_ratio == pytest.approx(0.85, abs=0.03)
