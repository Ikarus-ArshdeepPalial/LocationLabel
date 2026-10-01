import json
import sys

import pytest
import responses
from PIL import Image

import main
from common.addresses import read_jobs
from common.record import record_path
from user_images.config import Settings
from user_images.importer import ImageImporter
from user_images.tests.helpers import phone_exif, save_jpeg, textured, write_json

ADDRESS = "2910 Sussex Gardens Ln, Austin, TX 78748, USA"


def test_principal_point_follows_crop(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    pixels = textured(800, 600)
    pixels[:50] = 240   # UI bar at the top
    pixels[-20:] = 20   # UI bar at the bottom
    Image.fromarray(pixels).save(photos / "shot.png")  # screenshots are lossless PNGs
    write_json(photos / "camera.json", {"focal_px": 500, "pitch_deg": -8, "roll_deg": 0})

    ImageImporter(Settings(output_dir=tmp_path / "out", crop_ui=True), []).run(ADDRESS, photos)

    meta = json.loads(record_path(tmp_path / "out", ADDRESS).read_text())["streetview"]["images"][0]
    assert meta["crop"] == {"top": 50, "bottom": 20, "left": 0, "right": 0}
    assert (meta["width"], meta["height"]) == (800, 530)
    assert meta["intrinsics"] == {"fx": 500, "fy": 500, "cx": 400, "cy": 250}
    assert meta["metadata_sources"]["principal_point"] == "image_center_crop_adjusted"


def test_import_writes_stage2_structure_with_sources(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    save_jpeg(photos / "front.jpg", textured(400, 300), phone_exif(focal_35mm=26, heading=90))
    save_jpeg(photos / "side.jpg", textured(640, 480))
    write_json(photos / "side.json", {"fov_deg": 70, "pitch_deg": -12, "roll_deg": 0})
    write_json(photos / "camera.json", {"pitch_deg": -5, "roll_deg": 0, "camera_height_m": 1.4})

    status = ImageImporter(Settings(output_dir=tmp_path / "out"), []).run(ADDRESS, photos)

    section = json.loads(record_path(tmp_path / "out", ADDRESS).read_text())["streetview"]
    front, side = section["images"]
    assert status == "ok" and section["source"] == "user_supplied"
    assert front["metadata_sources"] == {
        "focal_px": "exif", "pitch_deg": "camera_json", "roll_deg": "camera_json", "heading_deg": "exif",
        "camera_height_m": "camera_json", "position": "exif", "principal_point": "image_center",
    }
    assert front["pano_lat"] == pytest.approx(30.1547, abs=1e-4) and front["heading"] == 90
    assert side["metadata_sources"]["focal_px"] == "sidecar" and side["pitch"] == -12
    assert side["pano_lat"] is None and side["metadata_sources"]["position"] == "unknown"
    assert side["fov"] == pytest.approx(70) and side["width"] == 640
    assert [p["pano_id"] for p in section["panos"]] == ["user:front"]
    assert (record_path(tmp_path / "out", ADDRESS).parent / "images" / "side.jpg").exists()


@responses.activate
def test_streetview_stage_skipped_with_images_dir(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    photos.mkdir()
    save_jpeg(photos / "a.jpg", textured(320, 240))
    write_json(photos / "camera.json", {"fov_deg": 65, "pitch_deg": 0, "roll_deg": 0})
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["main.py", "streetview", ADDRESS, "--images-dir", str(photos)])

    assert main.main() == 0

    assert len(responses.calls) == 0  # any HTTP request would have raised, since nothing is mocked
    section = json.loads(record_path(tmp_path / "out", ADDRESS).read_text())["streetview"]
    assert section["source"] == "user_supplied" and section["image_requests"] == 0


def test_csv_images_dir_column(tmp_path):
    csv_path = tmp_path / "batch.csv"
    csv_path.write_text('address,images_dir\n"1 Main St, Austin, TX",photos/one\n"2 Main St, Austin, TX",\n')

    jobs = read_jobs(csv_path)

    assert [(j.address, j.images_dir) for j in jobs] == [("1 Main St, Austin, TX", tmp_path / "photos/one"), ("2 Main St, Austin, TX", None)]
