import responses
from responses.matchers import query_param_matcher

from common.http import build_session
from groundTruth import arcgis
from groundTruth.arcgis import ArcGISError, layer_info, query_all, sample_features

LAYER_URL = "https://example.gov/arcgis/rest/services/Test/FeatureServer/0"


def metadata(max_record_count: int = 2) -> dict:
    return {"name": "Test Layer", "geometryType": "esriGeometryPoint", "maxRecordCount": max_record_count, "fields": [{"name": "OBJECTID", "type": "esriFieldTypeOID"}, {"name": "SLOPE", "type": "esriFieldTypeString"}]}


def feature(object_id: int, slope: str = "5.0") -> dict:
    return {"geometry": {"x": -97.0, "y": 30.0}, "attributes": {"OBJECTID": object_id, "SLOPE": slope}}


@responses.activate
def test_layer_info_reads_fields_and_page_size():
    responses.get(LAYER_URL, json=metadata(500))

    info = layer_info(build_session(0), LAYER_URL)

    assert info.reachable and info.max_record_count == 500
    assert info.fields == [{"name": "OBJECTID", "type": "OID"}, {"name": "SLOPE", "type": "String"}]


@responses.activate
def test_layer_info_reports_unreachable_on_non_json():
    responses.get(LAYER_URL, body="<html>Sorry. Page not served.</html>")

    info = layer_info(build_session(0), LAYER_URL)

    assert not info.reachable and "non-JSON" in info.error


@responses.activate
def test_sample_features_passes_where_and_limit():
    responses.get(f"{LAYER_URL}/query", json={"features": [feature(1)]}, match=[query_param_matcher({"resultRecordCount": "3", "where": "1=1"}, strict_match=False)])

    features = sample_features(build_session(0), LAYER_URL, n=3)

    assert len(features) == 1


@responses.activate
def test_query_all_paginates_until_a_short_page():
    responses.get(LAYER_URL, json=metadata(2))
    responses.get(f"{LAYER_URL}/query", json={"features": [feature(1), feature(2)]}, match=[query_param_matcher({"resultOffset": "0"}, strict_match=False)])
    responses.get(f"{LAYER_URL}/query", json={"features": [feature(3)]}, match=[query_param_matcher({"resultOffset": "2"}, strict_match=False)])

    features = query_all(build_session(0), LAYER_URL, delay=0)

    assert [f["attributes"]["OBJECTID"] for f in features] == [1, 2, 3]
    assert len(responses.calls) == 3  # 1 metadata + 2 query pages


@responses.activate
def test_query_all_stops_on_empty_page():
    responses.get(LAYER_URL, json=metadata(2))
    responses.get(f"{LAYER_URL}/query", json={"features": []})

    assert query_all(build_session(0), LAYER_URL, delay=0) == []


@responses.activate
def test_query_all_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(arcgis.time, "sleep", lambda seconds: None)
    responses.get(LAYER_URL, json=metadata(10))
    responses.get(f"{LAYER_URL}/query", status=503)
    responses.get(f"{LAYER_URL}/query", json={"features": [feature(1)]})

    features = query_all(build_session(0), LAYER_URL, delay=0, max_retries=2)

    assert len(features) == 1


@responses.activate
def test_query_all_raises_arcgis_error_from_service():
    responses.get(LAYER_URL, json={"error": {"code": 400, "message": "Invalid where clause"}})

    try:
        query_all(build_session(0), LAYER_URL)
        assert False, "expected ArcGISError"
    except ArcGISError as exc:
        assert "Invalid where clause" in str(exc)
