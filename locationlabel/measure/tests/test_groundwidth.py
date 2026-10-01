import numpy as np
import pytest

from measure.groundwidth import MIN_DEPRESSION_DEG, combine, project, two_method_width
from measure.tests.synthetic import CAMERA, SIZE


def strip_pixels(x0: float, x1: float, y0: float, y1: float, step: float = 0.05):
    """Pixels covered by a strip of ground, from x0 to x1 metres across and y0 to y1 metres ahead."""
    gx, gy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    world = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    rel = (world - CAMERA.center) @ CAMERA.rotation
    u, v = CAMERA.cx + CAMERA.fx * rel[:, 0] / rel[:, 2], CAMERA.cy + CAMERA.fy * rel[:, 1] / rel[:, 2]
    keep = (u >= 0) & (u < SIZE) & (v >= 0) & (v < SIZE)
    return np.unique(np.stack([u[keep], v[keep]], axis=1).round().astype(int), axis=0)


def ground_depth_map():
    v, u = np.mgrid[0:SIZE, 0:SIZE]
    return np.nan_to_num(CAMERA.ground_depth(u.ravel(), v.ravel()).reshape(SIZE, SIZE), nan=80.0).astype(np.float32)


def test_both_methods_read_a_known_strip_width_and_agree():
    surface, edge = strip_pixels(1.0, 2.6, 4.0, 11.0), strip_pixels(0.9, 1.0, 4.0, 11.0)  # a 1.6 m wide strip beside a curb edge
    strips = two_method_width(CAMERA, ground_depth_map(), surface[:, 0], surface[:, 1], edge[:, 0], edge[:, 1], 12.0)
    assert strips["ground_plane"].width_m == pytest.approx(1.6, abs=0.12) and strips["depth"].width_m == pytest.approx(1.6, abs=0.12)
    merged = combine(strips, (SIZE, SIZE))
    assert merged["agree"] and merged["certain"] and merged["width_m"] == pytest.approx(1.6, abs=0.12)


def test_without_depth_only_the_ground_plane_is_used_and_it_is_not_certain():
    surface, edge = strip_pixels(1.0, 2.6, 4.0, 11.0), strip_pixels(0.9, 1.0, 4.0, 11.0)
    merged = combine(two_method_width(CAMERA, None, surface[:, 0], surface[:, 1], edge[:, 0], edge[:, 1], 12.0), (SIZE, SIZE))
    assert merged["methods"].keys() == {"ground_plane"} and not merged["certain"] and merged["width_m"] == pytest.approx(1.6, abs=0.12)


def test_disagreeing_methods_are_not_certain_and_the_ground_plane_value_is_kept():
    surface, edge = strip_pixels(1.0, 2.6, 4.0, 11.0), strip_pixels(0.9, 1.0, 4.0, 11.0)
    strips = two_method_width(CAMERA, ground_depth_map() * 1.6, surface[:, 0], surface[:, 1], edge[:, 0], edge[:, 1], 40.0)
    merged = combine(strips, (SIZE, SIZE))
    assert not merged["agree"] and not merged["certain"] and merged["width_m"] == pytest.approx(strips["ground_plane"].width_m, abs=0.01)


def test_pixels_too_close_to_the_horizon_are_not_used():
    far = strip_pixels(-1.0, 1.0, 60.0, 90.0, step=0.5)  # about 2 degrees below the horizon
    got = project(CAMERA, None, far[:, 0], far[:, 1], 200.0, "ground")
    assert got is None or len(got[0]) == 0
    near = strip_pixels(-1.0, 1.0, 4.0, 8.0)
    assert len(project(CAMERA, None, near[:, 0], near[:, 1], 12.0, "ground")[0]) == len(near) > 0
    assert MIN_DEPRESSION_DEG >= 2.0
