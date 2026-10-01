import json
from pathlib import Path

import pytest
import responses

from addr_To_coord.config import Settings
from addr_To_coord.geocode import CENSUS_URL, GOOGLE_URL, Geocoder
from addr_To_coord.record import GeocodeResult, save_record
from common.cache import JsonCache

ADDRESS = "1600 Pennsylvania Ave NW, Washington, DC 20500"


def google_payload(country: str = "US", location_type: str = "ROOFTOP") -> dict:
    return {
        "status": "OK",
        "results": [{
            "formatted_address": "1600 Pennsylvania Avenue NW, Washington, DC 20500, USA",
            "geometry": {"location": {"lat": 38.8977, "lng": -77.0365}, "location_type": location_type},
            "address_components": [{"short_name": country, "types": ["country", "political"]}],
        }],
    }


def census_payload(matches: bool = True, number: str = "1600", lat: float = 38.8987, lng: float = -77.0352) -> dict:
    match = {
        "matchedAddress": f"{number} PENNSYLVANIA AVE NW, WASHINGTON, DC, 20500",
        "coordinates": {"x": lng, "y": lat},
        "tigerLine": {"side": "L", "tigerLineId": "76225813"},
        "addressComponents": {"fromAddress": "1600", "toAddress": "1648"},
    }
    return {"result": {"addressMatches": [match] if matches else []}}


def make_geocoder(tmp_path: Path, api_key: str | None = "test-key") -> Geocoder:
    settings = Settings(
        google_api_key=api_key,
        output_dir=tmp_path / "output",
        cache_path=tmp_path / "cache.sqlite",
        timeout=1,
        max_retries=0,
    )
    return Geocoder(settings, JsonCache(settings.cache_path))


@responses.activate
def test_google_success(tmp_path):
    responses.get(GOOGLE_URL, json=google_payload())

    result = make_geocoder(tmp_path).geocode(ADDRESS)

    assert result.status == "ok"
    assert result.provider == "google"
    assert result.match_quality == "ROOFTOP"
    assert result.partial_match is False
    assert (result.lat, result.lng) == (38.8977, -77.0365)


@responses.activate
def test_google_failure_falls_back_to_census(tmp_path):
    responses.get(GOOGLE_URL, status=500)
    responses.get(CENSUS_URL, json=census_payload())

    result = make_geocoder(tmp_path).geocode(ADDRESS)

    assert result.status == "ok"
    assert result.provider == "census"
    assert result.match_quality == "RANGE_INTERPOLATED"
    assert (result.lat, result.lng) == (38.8987, -77.0352)


@responses.activate
def test_missing_api_key_uses_census(tmp_path):
    responses.get(CENSUS_URL, json=census_payload())

    result = make_geocoder(tmp_path, api_key=None).geocode(ADDRESS)

    assert result.provider == "census"
    assert all(call.request.url.startswith(CENSUS_URL) for call in responses.calls)


def census_for(number: str, **payload) -> responses.BaseResponse:
    street = f"{number} PENNSYLVANIA AVE NW, WASHINGTON, DC, 20500"
    return responses.get(
        CENSUS_URL,
        json=census_payload(number=number, **payload),
        match=[responses.matchers.query_param_matcher({"address": street}, strict_match=False)],
    )


@responses.activate
def test_census_saves_street_segment(tmp_path):
    responses.get(
        CENSUS_URL,
        json=census_payload(),
        match=[responses.matchers.query_param_matcher({"address": ADDRESS}, strict_match=False)],
    )
    census_for("1600")
    census_for("1648", lat=38.8990, lng=-77.0340)

    result = make_geocoder(tmp_path, api_key=None).geocode(ADDRESS)

    assert result.street_segment == {
        "tiger_line_id": "76225813",
        "side": "L",
        "from": {"address_number": "1600", "lat": 38.8987, "lng": -77.0352},
        "to": {"address_number": "1648", "lat": 38.8990, "lng": -77.0340},
    }


@responses.activate
def test_non_us_result_rejected(tmp_path):
    responses.get(GOOGLE_URL, json=google_payload(country="FR"))

    result = make_geocoder(tmp_path).geocode("10 Rue de Rivoli, Paris")

    assert result.status == "error"
    assert "outside the United States" in result.error
    assert result.lat is None


@responses.activate
def test_no_result_found(tmp_path):
    responses.get(GOOGLE_URL, json={"status": "ZERO_RESULTS", "results": []})
    responses.get(CENSUS_URL, json=census_payload(matches=False))

    result = make_geocoder(tmp_path).geocode("nowhere at all")

    assert result.status == "error"
    assert result.error == "No result found"


@responses.activate
def test_cache_hit_skips_api(tmp_path):
    responses.get(GOOGLE_URL, json=google_payload())
    geocoder = make_geocoder(tmp_path)

    first = geocoder.geocode(ADDRESS)
    second = geocoder.geocode("  1600 pennsylvania ave nw washington dc 20500 ")

    assert len(responses.calls) == 1
    assert (second.lat, second.lng, second.provider) == (first.lat, first.lng, "google")


def test_save_record_preserves_later_stage_keys(tmp_path):
    path = save_record(GeocodeResult(input_address=ADDRESS, lat=1.0, lng=2.0), tmp_path)
    record = json.loads(path.read_text())
    record["streetview"] = {"images": []}
    path.write_text(json.dumps(record))

    save_record(GeocodeResult(input_address=ADDRESS, lat=3.0, lng=4.0), tmp_path)

    updated = json.loads(path.read_text())
    assert updated["streetview"] == {"images": []}
    assert updated["lat"] == 3.0


@responses.activate
def test_rejected_key_is_reported_once_and_google_skipped(tmp_path, caplog):
    responses.get(GOOGLE_URL, json={"status": "REQUEST_DENIED", "error_message": "billing disabled"})
    responses.get(CENSUS_URL, json=census_payload())
    geocoder = make_geocoder(tmp_path)

    geocoder.geocode(ADDRESS)
    geocoder.geocode("350 Fifth Avenue, New York, NY 10118")

    google_calls = [c for c in responses.calls if c.request.url.startswith(GOOGLE_URL)]
    assert len(google_calls) == 1
    assert caplog.text.count("rejected the API key") == 1
