import json

import pytest

from measure.limits import DEFAULTS, MEASURES, Limit, evaluate, from_base, load_limits, save_limits, to_base


def rule(measure="driveway_running_slope", limit=10.0, unit="%", margin=0.5, **kwargs) -> Limit:
    return Limit("test", measure, limit, unit, margin, **kwargs)


def exceeded(limits, measure, base):
    return [c.exceeded for c in evaluate(limits, measure, base, "x")]


def test_a_value_over_the_limit_by_more_than_the_margin_is_exceeded():
    assert exceeded([rule()], "driveway_running_slope", 10.6) == [True]


def test_over_the_limit_by_less_than_the_margin_is_not_a_message():
    assert exceeded([rule()], "driveway_running_slope", 10.4) == [False]
    assert exceeded([rule()], "driveway_running_slope", 10.5) == [False]  # exactly the margin is still allowed
    assert exceeded([rule()], "driveway_running_slope", 9.0) == [False]


def test_the_margin_is_whatever_the_user_sets():
    assert exceeded([rule(margin=0.0)], "driveway_running_slope", 10.1) == [True]
    assert exceeded([rule(margin=2.0)], "driveway_running_slope", 11.9) == [False]


def test_a_minimum_is_exceeded_when_the_value_is_too_low():
    minimum = Limit("sign", "stop_sign_bottom_height", 7.0, "ft", 0.2)
    assert exceeded([minimum], "stop_sign_bottom_height", 1.9) == [True]  # 1.9 m is 6.2 ft
    assert exceeded([minimum], "stop_sign_bottom_height", 2.1) == [False]  # 6.9 ft: short by 0.1 ft, inside the 0.2 ft margin
    assert exceeded([minimum], "stop_sign_bottom_height", 2.3) == [False]


def test_a_rule_in_degrees_is_compared_in_the_same_way():
    in_degrees = Limit("d", "driveway_running_slope", 5.7, "°", 0.3)
    assert exceeded([in_degrees], "driveway_running_slope", to_base(6.2, "°")) == [True]
    assert exceeded([in_degrees], "driveway_running_slope", to_base(5.9, "°")) == [False]


def test_units_convert_both_ways():
    assert to_base(10, "%") == 10
    assert to_base(45, "°") == pytest.approx(100)
    assert to_base(7, "ft") == pytest.approx(2.1336)
    assert from_base(2.1336, "ft") == pytest.approx(7)
    assert from_base(to_base(5.7, "°"), "°") == pytest.approx(5.7)


def test_disabled_and_other_measures_are_ignored():
    assert evaluate([rule(enabled=False)], "driveway_running_slope", 20.0, "x") == []
    assert evaluate([rule()], "ramp_running_slope", 20.0, "x") == []


def test_the_message_names_the_limit_and_whether_it_is_certain():
    check = evaluate([rule()], "driveway_running_slope", 12.0, "x", certain=True)[0]
    assert check.message == "12.0 % · limit exceeded: max 10.0 %" and check.value_text == "6.8°"  # a slope leads with degrees
    assert evaluate([rule()], "driveway_running_slope", 12.0, "x", certain=False)[0].message == "12.0 % · may exceed the limit: max 10.0 %"
    low = evaluate([Limit("s", "stop_sign_bottom_height", 7, "ft", 0.2)], "stop_sign_bottom_height", 1.5, "x")[0]
    assert low.message == "below the limit: min 7.0 ft"


def test_limits_round_trip_and_bad_entries_are_skipped(tmp_path):
    path = tmp_path / "limits.json"
    save_limits(path, DEFAULTS)
    assert [r.name for r in load_limits(path)] == [r.name for r in DEFAULTS]

    path.write_text(json.dumps([{"name": "ok", "measure": "pole_height", "limit": 9, "unit": "m"}, {"name": "bad measure", "measure": "nope", "limit": 1, "unit": "%"},
                                {"name": "bad unit", "measure": "pole_height", "limit": 1, "unit": "%"}, {"measure": "pole_height"}]), encoding="utf-8")
    assert [r.name for r in load_limits(path)] == ["ok"]
    assert load_limits(tmp_path / "missing.json") == [] and load_limits(None) == []


def test_every_default_uses_a_known_measure_and_unit():
    for default in DEFAULTS:
        assert default.measure in MEASURES and default.limit > 0
