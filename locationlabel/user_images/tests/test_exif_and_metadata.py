from math import atan, degrees

import pytest

from common.settings import ROOT_DIR
from user_images.calibration import Calibration
from user_images.config import Settings
from user_images.exif import ExifInfo, load_oriented, parse_exif
from user_images.metadata import resolve_camera
from user_images.tests.helpers import FakeCalibrator, phone_exif, save_jpeg, textured

SETTINGS = Settings(output_dir=ROOT_DIR / "output")


def test_exif_focal_gps_heading(tmp_path):
    path = save_jpeg(tmp_path / "a.jpg", textured(400, 300), phone_exif(focal_35mm=26))

    image, raw = load_oriented(path)
    info = parse_exif(raw, image.shape[1], image.shape[0])

    assert info.focal_px == pytest.approx(26 * 500 / 43.2666, rel=1e-4)  # 400x300 image has a 500 px diagonal
    assert degrees(2 * atan(200 / info.focal_px)) == pytest.approx(67.3, abs=0.1)
    assert info.lat == pytest.approx(30 + 9 / 60 + 16.8 / 3600)
    assert info.lng == pytest.approx(-(97 + 51 / 60 + 18 / 3600))
    assert info.heading_deg == pytest.approx(123.5)
    assert info.capture_date == "2026:05:01 10:00:00"


def test_exif_orientation_rotates_pixels(tmp_path):
    pixels = textured(400, 300)
    pixels[:20, :20] = (255, 0, 0)  # marker in the stored top-left corner
    path = save_jpeg(tmp_path / "rotated.jpg", pixels, phone_exif(orientation=6))

    image, raw = load_oriented(path)

    assert image.shape[:2] == (400, 300)  # orientation 6 = rotate 90 degrees clockwise for display
    assert image[5, -5, 0] > 200 and image[5, -5, 1] < 60  # the marker is now top-right
    assert parse_exif(raw, 300, 400).orientation == 6


def resolve(sidecar=None, folder=None, exif=None, calibrators=(), settings=SETTINGS, width=800):
    image = textured(width, 600)
    return resolve_camera(image, sidecar or {}, folder or {}, exif or ExifInfo(), settings, list(calibrators))


def test_sidecar_beats_camera_json_beats_exif():
    exif = ExifInfo(focal_px=900, heading_deg=45, lat=30.0, lng=-97.0)
    folder = {"fov_deg": 90, "pitch_deg": -5, "camera_height_m": 1.6, "heading_deg": 10}
    sidecar = {"focal_px": 500, "pitch_deg": -10, "roll_deg": 0}

    with_sidecar = resolve(sidecar, folder, exif)
    only_folder = resolve({"roll_deg": 0}, folder, exif)

    assert (with_sidecar.focal_px.value, with_sidecar.focal_px.source) == (500, "sidecar")
    assert (with_sidecar.pitch_deg.value, with_sidecar.pitch_deg.source) == (-10, "sidecar")
    assert (with_sidecar.camera_height_m.value, with_sidecar.camera_height_m.source) == (1.6, "camera_json")
    assert (with_sidecar.heading_deg.value, with_sidecar.heading_deg.source) == (10, "camera_json")
    assert (with_sidecar.position.value, with_sidecar.position.source) == ((30.0, -97.0), "exif")
    assert only_folder.focal_px.value == pytest.approx(400) and only_folder.focal_px.source == "camera_json"  # 90 deg on 800 px
    assert only_folder.pitch_deg.source == "camera_json"


def test_exif_used_when_no_sidecar():
    camera = resolve({"pitch_deg": 0, "roll_deg": 0}, exif=ExifInfo(focal_px=700, heading_deg=200))

    assert (camera.focal_px.value, camera.focal_px.source) == (700, "exif")
    assert (camera.heading_deg.value, camera.heading_deg.source) == (200, "exif")
    assert camera.position.source == "unknown" and camera.position.value is None


def test_cli_camera_height_overrides_everything():
    camera = resolve(folder={"camera_height_m": 1.6}, settings=Settings(output_dir=ROOT_DIR, camera_height_override=2.1))

    assert (camera.camera_height_m.value, camera.camera_height_m.source) == (2.1, "cli")


def test_fallback_chain_uses_first_working_calibrator():
    failing = FakeCalibrator("geocalib", None)
    working = FakeCalibrator("lines", Calibration(None, pitch_deg=4.0, roll_deg=1.5))

    camera = resolve(exif=ExifInfo(focal_px=650), calibrators=[failing, working])

    assert failing.calls == working.calls == 1
    assert camera.calibrator == "lines"
    assert (camera.pitch_deg.value, camera.pitch_deg.source) == (4.0, "estimated")
    assert (camera.roll_deg.value, camera.roll_deg.source) == (1.5, "estimated")
    assert (camera.focal_px.value, camera.focal_px.source) == (650, "exif")  # known focal is kept over estimation


def test_estimated_focal_used_when_nothing_else():
    camera = resolve(calibrators=[FakeCalibrator("geocalib", Calibration(520.0, -3.0, 0.0))])

    assert (camera.focal_px.value, camera.focal_px.source) == (520.0, "estimated")


def test_defaults_when_everything_fails():
    camera = resolve(calibrators=[FakeCalibrator("geocalib", None), FakeCalibrator("lines", None)])

    phone = SETTINGS.defaults
    assert camera.calibrator is None
    assert degrees(2 * atan(400 / camera.focal_px.value)) == pytest.approx(phone.fov_deg)
    assert camera.focal_px.source == camera.pitch_deg.source == camera.heading_deg.source == "default"
    assert (camera.camera_height_m.value, camera.camera_height_m.source) == (phone.camera_height_m, "default")


def test_calibrator_skipped_when_metadata_complete():
    calibrator = FakeCalibrator("geocalib", Calibration(1.0, 1.0, 1.0))

    resolve({"fov_deg": 70, "pitch_deg": 0, "roll_deg": 0}, calibrators=[calibrator])

    assert calibrator.calls == 0
