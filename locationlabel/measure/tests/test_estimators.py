import numpy as np
import pytest

from common.settings import ROOT_DIR
from label_objs.config import CLASSES
from measure.config import Settings
from measure.depth import depth_scale
from measure.estimators import (
    ROAD_IDS,
    curb_flat_ground,
    curb_point_cloud,
    pole_base_problem,
    sign_flat_ground,
    sign_point_cloud,
    sign_reference_scale,
)
from measure.geometry import Camera
from measure.tests.synthetic import CAMERA, SIGN_BOTTOM, project, scene

SETTINGS = Settings(output_dir=ROOT_DIR / "output")


def sign_inputs():
    semantic, depth, sign, pole = scene()
    ys, xs = np.nonzero(sign)
    pole_ys, _ = np.nonzero(pole)
    box = (xs.min(), ys.min(), xs.max(), ys.max())
    return semantic, depth, sign, pole, box, [320, int(pole_ys.max())], [320.0, float(ys.max())]


def test_sign_flat_ground_height():
    _, _, _, _, _, base, bottom = sign_inputs()

    estimate = sign_flat_ground(CAMERA, base, bottom, SETTINGS)

    assert estimate.value == pytest.approx(SIGN_BOTTOM, abs=0.04)
    assert estimate.distance_m == pytest.approx(8, abs=0.1)
    assert 0 < estimate.uncertainty < 0.3


def test_sign_reference_scale_height():
    _, _, _, _, box, base, bottom = sign_inputs()

    estimate = sign_reference_scale(CAMERA, box, base, bottom, SETTINGS)

    assert estimate.value == pytest.approx(SIGN_BOTTOM, rel=0.05)


def test_sign_reference_scale_corrects_camera_pitch():
    camera = Camera(320, 320, 320, 320, heading=0, pitch=12, height=2.5)
    middle = SIGN_BOTTOM + 0.381  # an octagon is widest at mid-height
    left, _ = project(camera, [-0.381, 8, middle])
    right, _ = project(camera, [0.381, 8, middle])
    _, top = project(camera, [0, 8, SIGN_BOTTOM + 0.762])
    _, bottom = project(camera, [0, 8, SIGN_BOTTOM])
    _, base = project(camera, [0, 8, 0])

    box = (left + 0.5, top, right - 0.5, bottom)  # pixel-index convention: width = x2 - x1 + 1
    estimate = sign_reference_scale(camera, box, [320, base], [320, bottom], SETTINGS)

    assert estimate.value == pytest.approx(SIGN_BOTTOM, rel=0.03)


def test_sign_reference_scale_rejects_foreshortened_sign():
    estimate = sign_reference_scale(CAMERA, (300, 300, 310, 330), [305, 420], [305, 330], SETTINGS)

    assert not estimate.applicable and "foreshortened" in estimate.reason


def test_sign_point_cloud_height():
    _, depth, sign, pole, _, base, bottom = sign_inputs()

    estimate = sign_point_cloud(CAMERA, depth, sign, pole, base, bottom)

    assert estimate.value == pytest.approx(SIGN_BOTTOM, abs=0.04)


def test_pole_base_visibility():
    semantic, *_ = sign_inputs()
    _, _, _, _, _, base, _ = sign_inputs()
    assert pole_base_problem(semantic, base) is None

    occluded = semantic.copy()
    occluded[base[1] + 1:base[1] + 6, :] = CLASSES.index("car")
    assert pole_base_problem(occluded, base) == "pole base occluded"
    assert "edge" in pole_base_problem(semantic, [320, 639])


def curb_scene(curb_distance: float = 6.0, curb_height: float = 0.15):
    """Level camera looking at a straight curb across the view: road, then a raised sidewalk."""
    size = 640
    v, u = np.mgrid[0:size, 0:size]
    rays = CAMERA.rays_world(u.ravel(), v.ravel())
    down = rays[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_road = -CAMERA.height / down
        t_side = -(CAMERA.height - curb_height) / down
    road = (down < 0) & (t_road * rays[:, 1] < curb_distance)
    side = (down < 0) & ~road & (t_side * rays[:, 1] >= curb_distance)
    face = (down < 0) & ~road & ~side

    depth = np.full(size * size, 80.0)
    depth[road], depth[side] = t_road[road], t_side[side]
    depth[face] = curb_distance  # z-depth of the vertical curb face for a level camera
    semantic = np.zeros(size * size, np.uint8)
    semantic[road], semantic[side], semantic[face] = CLASSES.index("road"), CLASSES.index("sidewalk"), CLASSES.index("curb")

    bottom_row = int(np.ceil(CAMERA.cy + CAMERA.fy * CAMERA.height / curb_distance))
    curb_line = [[x, bottom_row] for x in range(40, 600, 20)]
    return semantic.reshape(size, size), depth.reshape(size, size).astype(np.float32), curb_line


def test_curb_flat_ground_height():
    semantic, _, curb_line = curb_scene()

    estimate = curb_flat_ground(CAMERA, semantic, curb_line, SETTINGS)

    assert estimate.value == pytest.approx(0.15, abs=0.02)


def test_curb_point_cloud_height():
    semantic, depth, curb_line = curb_scene()

    estimate = curb_point_cloud(CAMERA, depth, semantic, curb_line, SETTINGS)

    assert estimate.value == pytest.approx(0.15, abs=0.01)


def test_depth_scale_correction():
    semantic, depth, _ = curb_scene()

    scale = depth_scale(depth * 1.25, CAMERA, np.isin(semantic, ROAD_IDS), max_distance=15)

    assert scale == pytest.approx(0.8, rel=1e-3)
