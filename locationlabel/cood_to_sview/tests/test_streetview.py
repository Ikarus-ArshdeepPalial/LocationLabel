import io
import json
from pathlib import Path

import pytest
import responses
from PIL import Image
from responses.matchers import query_param_matcher

from common.cache import JsonCache
from common.http import build_session
from common.record import record_path
from cood_to_sview.config import Settings
from cood_to_sview.geo import bearing, destination, distance
from cood_to_sview.provider import GoogleStreetViewProvider
from cood_to_sview.streetview import StreetViewStage
from cood_to_sview.target import resolve_target

META_URL = GoogleStreetViewProvider.METADATA_URL
IMAGE_URL = GoogleStreetViewProvider.IMAGE_URL
ADDRESS = "2910 Sussex Gardens Ln, Austin, TX 78748"

STAGE1_RECORD = {
    "input_address": ADDRESS,
    "formatted_address": "2910 SUSSEX GARDENS LN, AUSTIN, TX, 78748",
    "lat": 30.154712,
    "lng": -97.855083,
    "provider": "census",
    "match_quality": "RANGE_INTERPOLATED",
    "street_segment": {
        "tiger_line_id": "63991001",
        "side": "L",
        "from": {"address_number": "2916", "lat": 30.154990, "lng": -97.855080},
        "to": {"address_number": "2900", "lat": 30.154349, "lng": -97.855088},
    },
    "status": "ok",
    "error": None,
}


def jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buffer, "JPEG")
    return buffer.getvalue()


def pano_payload(pano_id: str = "MAINPANO123", lat: float = 30.15471, lng: float = -97.85516) -> dict:
    return {"status": "OK", "pano_id": pano_id, "location": {"lat": lat, "lng": lng}, "date": "2023-05"}


def mock_metadata(radius: int, payload: dict) -> None:
    responses.get(META_URL, json=payload, match=[query_param_matcher({"radius": str(radius)}, strict_match=False)])


def make_stage(tmp_path: Path, **options) -> tuple[StreetViewStage, GoogleStreetViewProvider]:
    settings = Settings(
        google_api_key="test-key",
        output_dir=tmp_path / "output",
        cache_path=tmp_path / "cache.sqlite",
        timeout=1,
        max_retries=0,
    )
    session = build_session(0)
    provider = GoogleStreetViewProvider("test-key", session, 1)
    return StreetViewStage(settings, provider, JsonCache(settings.cache_path), session, **options), provider


@pytest.fixture
def stage1_record(tmp_path) -> Path:
    path = record_path(tmp_path / "output", ADDRESS)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(STAGE1_RECORD))
    return path


def test_bearing_known_pairs():
    assert bearing(0, 0, 1, 0) == pytest.approx(0)
    assert bearing(0, 0, 0, 1) == pytest.approx(90)
    assert bearing(0, 0, -1, 0) == pytest.approx(180)
    assert bearing(0, 0, 0, -1) == pytest.approx(270)
    assert bearing(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(148.1, abs=0.2)  # London -> Paris


def test_destination_round_trip():
    lat, lng = destination(30.0, -97.0, 60, 20)
    assert distance(30.0, -97.0, lat, lng) == pytest.approx(20, abs=0.01)
    assert bearing(30.0, -97.0, lat, lng) == pytest.approx(60, abs=0.01)


@pytest.mark.parametrize("side, east_of_original", [("L", True), ("R", False)])
def test_side_offset_direction(side, east_of_original):
    # The segment runs south, so the left side is east and the right side is west.
    record = {**STAGE1_RECORD, "street_segment": {**STAGE1_RECORD["street_segment"], "side": side}}

    target = resolve_target(record, offset_m=15)

    assert target.method == "census_side_offset"
    assert target.street_bearing == pytest.approx(180, abs=1)
    assert (target.adjusted.lng > target.original.lng) is east_of_original
    assert distance(target.original.lat, target.original.lng, target.adjusted.lat, target.adjusted.lng) == pytest.approx(15, abs=0.1)


@responses.activate
def test_pano_found_saves_images_and_metadata(tmp_path, stage1_record):
    mock_metadata(50, pano_payload())
    mock_metadata(15, pano_payload("NEIGHBOR01", lat=30.15485))
    responses.get(IMAGE_URL, body=jpeg_bytes(), content_type="image/jpeg")
    stage, provider = make_stage(tmp_path)

    assert stage.run(ADDRESS) == "ok"

    record = json.loads(stage1_record.read_text())
    streetview = record["streetview"]
    assert record["lat"] == STAGE1_RECORD["lat"]
    assert [p["pano_id"] for p in streetview["panos"]] == ["MAINPANO123", "NEIGHBOR01"]
    assert len(streetview["images"]) == 6 + 2 == provider.image_requests
    image = streetview["images"][0]
    assert image["intrinsics"] == {"fx": pytest.approx(320), "fy": pytest.approx(320), "cx": 320, "cy": 320}
    assert (stage1_record.parent / "images" / image["filename"]).exists()
    assert (stage1_record.parent / "contact_sheet.jpg").exists()


@responses.activate
def test_pano_found_only_at_larger_radius(tmp_path, stage1_record):
    mock_metadata(50, {"status": "ZERO_RESULTS"})
    mock_metadata(100, pano_payload())
    mock_metadata(15, {"status": "ZERO_RESULTS"})
    responses.get(IMAGE_URL, body=jpeg_bytes())
    stage, _ = make_stage(tmp_path)

    assert stage.run(ADDRESS) == "ok"

    panos = json.loads(stage1_record.read_text())["streetview"]["panos"]
    assert [(p["pano_id"], p["search_radius_m"]) for p in panos] == [("MAINPANO123", 100)]


@responses.activate
def test_no_imagery(tmp_path, stage1_record):
    responses.get(META_URL, json={"status": "ZERO_RESULTS"})
    stage, provider = make_stage(tmp_path)

    assert stage.run(ADDRESS) == "no_imagery"
    assert provider.image_requests == 0
    assert json.loads(stage1_record.read_text())["streetview"]["status"] == "no_imagery"


@responses.activate
def test_cache_hit_skips_metadata_and_downloads(tmp_path, stage1_record):
    mock_metadata(50, pano_payload())
    mock_metadata(15, {"status": "ZERO_RESULTS"})
    responses.get(IMAGE_URL, body=jpeg_bytes())
    make_stage(tmp_path)[0].run(ADDRESS)
    calls_after_first_run = len(responses.calls)

    stage, provider = make_stage(tmp_path)
    assert stage.run(ADDRESS) == "ok"

    new_calls = [c.request.url for c in responses.calls[calls_after_first_run:]]
    assert provider.image_requests == 0
    assert all("radius=15" in url for url in new_calls)  # only the uncached "no pano" neighbor lookups


@responses.activate
def test_dry_run_makes_no_image_requests(tmp_path, stage1_record):
    mock_metadata(50, pano_payload())
    mock_metadata(15, {"status": "ZERO_RESULTS"})
    stage, provider = make_stage(tmp_path, dry_run=True)

    assert stage.run(ADDRESS) == "dry_run"
    assert provider.image_requests == 0
    assert not any(c.request.url.startswith(IMAGE_URL + "?") for c in responses.calls)
    assert "streetview" not in json.loads(stage1_record.read_text())
