import pytest

from groundTruth.fields import (
    convert_unit,
    extract_attribute,
    is_plausible,
    map_attributes,
    parse_compass_to_degrees,
    parse_date_string,
    parse_epoch_ms,
    parse_mutcd_code,
    parse_sign_size_inches,
    to_number,
)
from groundTruth.sources import FieldMapping


@pytest.mark.parametrize("raw, expected", [("5.0", 5.0), ("", None), (None, None), ("N/A", None), (7, 7.0), (3.5, 3.5)])
def test_to_number(raw, expected):
    assert to_number(raw) == expected


@pytest.mark.parametrize("unit, value, expected", [
    ("percent", 5.0, 5.0),
    ("ratio", 0.05, 5.0),
    ("inch", 45.5, 45.5),
    ("foot", 5.0, 60.0),
    ("cm", 2.54, pytest.approx(1.0)),
    ("mm", 25.4, pytest.approx(1.0)),
    ("meter", 1.0, pytest.approx(39.37, abs=0.01)),
])
def test_convert_unit(unit, value, expected):
    assert convert_unit(value, unit) == expected


def test_convert_unit_no_unit_passes_through():
    assert convert_unit(12.0, None) == 12.0


def test_parse_mutcd_code_extracts_leading_code():
    assert parse_mutcd_code("R1-1 (stop sign)") == "R1-1"
    assert parse_mutcd_code("W3-1 (stop sign ahead sign)") == "W3-1"
    assert parse_mutcd_code("D3-103X") == "D3-103X"
    assert parse_mutcd_code("") is None
    assert parse_mutcd_code(None) is None


@pytest.mark.parametrize("raw, expected", [('30" x 30"', 30.0), ("24x24", 24.0), ("VariesX9", 9.0), ("", None), (None, None)])
def test_parse_sign_size_inches(raw, expected):
    assert parse_sign_size_inches(raw) == expected


@pytest.mark.parametrize("raw, expected", [("NORTH", 0), ("N", 0), ("SE", 135), ("southwest", 225), ("", None), (None, None), ("XX", None)])
def test_parse_compass_to_degrees(raw, expected):
    assert parse_compass_to_degrees(raw) == expected


def test_parse_epoch_ms_rejects_sentinel_dates():
    assert parse_epoch_ms(1614729600000).year == 2021
    assert parse_epoch_ms(-2035324800000, min_year=1980) is None  # ~1905 sentinel for "unknown"
    assert parse_epoch_ms(None) is None


def test_parse_date_string():
    assert parse_date_string("9/26/2017 13:03", "%m/%d/%Y %H:%M").year == 2017
    assert parse_date_string("not a date", "%m/%d/%Y %H:%M") is None
    assert parse_date_string(None, "%m/%d/%Y") is None


def test_extract_attribute_uses_fallback_when_primary_is_null():
    mapping = FieldMapping(source_field="RP_Slope", fallback=["RP_Slope_LT", "RP_Slope_RT"], unit="percent")
    assert extract_attribute({"RP_Slope": None, "RP_Slope_LT": "8.0"}, mapping) == 8.0
    assert extract_attribute({"RP_Slope": "5.0", "RP_Slope_LT": "8.0"}, mapping) == 5.0
    assert extract_attribute({"RP_Slope": None, "RP_Slope_LT": None}, mapping) is None


def test_extract_attribute_plain_string_field_has_no_unit_or_parse():
    mapping = FieldMapping(source_field="SIGNTYPE")
    assert extract_attribute({"SIGNTYPE": "R1-1"}, mapping) == "R1-1"


def test_extract_attribute_with_parser():
    mapping = FieldMapping(source_field="TYPE_", parse="mutcd_code")
    assert extract_attribute({"TYPE_": "R1-1 (stop sign)"}, mapping) == "R1-1"


def test_map_attributes_applies_every_mapped_field():
    mapping = {
        "running_slope_pct": FieldMapping(source_field="SLOPE", unit="percent"),
        "sign_code": FieldMapping(source_field="TYPE_", parse="mutcd_code"),
    }
    result = map_attributes({"SLOPE": "5.0", "TYPE_": "R1-1"}, mapping)
    assert result == {"running_slope_pct": 5.0, "sign_code": "R1-1"}


@pytest.mark.parametrize("attrs", [{"running_slope_pct": 5.0}, {"running_slope_pct": None}, {"sign_size_in": 30}])
def test_is_plausible_accepts_good_values(attrs):
    ok, category, detail = is_plausible("curb_ramp" if "running_slope_pct" in attrs else "stop_sign", attrs)
    assert ok and category is None and detail is None


def test_is_plausible_rejects_out_of_range_slope():
    ok, category, detail = is_plausible("curb_ramp", {"running_slope_pct": 45.0})
    assert not ok and category == "running_slope_pct" and "45.0" in detail


def test_is_plausible_rejects_negative_slope():
    ok, category, _ = is_plausible("curb_ramp", {"cross_slope_pct": -5.0})
    assert not ok and category == "cross_slope_pct"


def test_is_plausible_rejects_non_mutcd_sign_size():
    ok, category, _ = is_plausible("stop_sign", {"sign_size_in": 27.0})
    assert not ok and category == "sign_size_in"


@pytest.mark.parametrize("size", [18, 24, 30, 36, 48])
def test_is_plausible_accepts_every_mutcd_stop_sign_size(size):
    ok, _, _ = is_plausible("stop_sign", {"sign_size_in": float(size)})
    assert ok


def test_is_plausible_only_checks_stop_sign_size_set_for_stop_signs():
    ok, _, _ = is_plausible("sign_other", {"sign_size_in": 27.0})
    assert ok  # non-standard sizes are common for non-R1-1 signs
