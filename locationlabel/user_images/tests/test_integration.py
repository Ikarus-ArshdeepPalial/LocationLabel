from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_geocalib_agrees_with_vanishing_points_on_real_photo():
    import ultralytics

    from user_images.calibration import GeoCalibCalibrator, LineCalibrator
    from user_images.exif import load_oriented

    image, _ = load_oriented(Path(ultralytics.__file__).parent / "assets" / "bus.jpg")

    learned = GeoCalibCalibrator().estimate(image, None)
    lines = LineCalibrator(default_fov_deg=60).estimate(image, learned.focal_px)

    assert 0.3 * image.shape[1] < learned.focal_px < 5 * image.shape[1]
    assert learned.pitch_deg == pytest.approx(lines.pitch_deg, abs=4)  # same sign convention and similar value
    assert learned.roll_deg == pytest.approx(lines.roll_deg, abs=3)
