from math import cos, radians, sin

import numpy as np
import pytest

from user_images.calibration import LineCalibrator, angles_from_up
from user_images.crop import Crop, detect_ui_bars
from user_images.importer import rotate_upright
from user_images.tests.helpers import textured, vertical_lines_image

FOCAL = 400.0
FOV = 2 * np.degrees(np.arctan(320 / FOCAL))


@pytest.mark.parametrize("pitch", [10.0, -15.0, 0.0])
def test_line_calibrator_recovers_pitch(pitch):
    result = LineCalibrator(default_fov_deg=FOV).estimate(vertical_lines_image(pitch, focal=FOCAL), None)

    assert result.pitch_deg == pytest.approx(pitch, abs=1.0)
    assert result.roll_deg == pytest.approx(0, abs=1.0)
    assert result.focal_px is None


def test_line_calibrator_recovers_roll_and_rotation_undoes_it():
    image = vertical_lines_image(5.0, roll_deg=6.0, focal=FOCAL)
    calibrator = LineCalibrator(default_fov_deg=FOV)

    rolled = calibrator.estimate(image, FOCAL)
    upright = calibrator.estimate(rotate_upright(image, rolled.roll_deg), FOCAL)

    assert rolled.roll_deg == pytest.approx(6.0, abs=1.0)
    assert upright.roll_deg == pytest.approx(0, abs=0.7)
    assert upright.pitch_deg == pytest.approx(5.0, abs=1.5)


def test_line_calibrator_gives_up_without_lines():
    assert LineCalibrator(default_fov_deg=FOV).estimate(np.full((300, 300, 3), 128, np.uint8), None) is None


def test_geocalib_up_vector_convention():
    # GeoCalib's Gravity.from_rp(roll=0, pitch=p) is (0, -cos p, sin p): the camera-frame up vector.
    pitch, roll = angles_from_up(np.array([0.0, -cos(radians(12)), sin(radians(12))]))
    assert pitch == pytest.approx(12) and roll == pytest.approx(0)


def test_detect_ui_bars():
    image = textured(400, 300)
    image[:40] = 245   # flat top toolbar
    image[-25:] = 30   # flat bottom strip

    assert detect_ui_bars(image) == Crop(top=40, bottom=25, left=0, right=0)


def test_crop_union_and_apply():
    crop = Crop(top=10, bottom=0, left=5, right=0).union(Crop(top=4, bottom=12, left=0, right=0))

    assert crop == Crop(10, 12, 5, 0)
    assert crop.apply(np.zeros((100, 200, 3))).shape == (78, 195, 3)
