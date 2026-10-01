import json

import pytest

from measure.config import Settings
from measure.limits import Limit, evaluate, save_limits
from measure.stage import RED, MeasurementStage, limit_label, sign_label
from measure.tests import synthetic
from measure.tests.synthetic import ADDRESS, SIGN_BOTTOM, FakeDepth, write_address


def run_with(tmp_path, limits):
    path, depth = write_address(tmp_path)
    limits_file = tmp_path / "limits.json"
    save_limits(limits_file, limits)
    section = MeasurementStage(Settings(output_dir=tmp_path, limits_file=limits_file if limits is not None else None), FakeDepth(depth)).run(ADDRESS)
    return path, section


def test_no_limits_file_means_no_limit_section(tmp_path):
    path, depth = write_address(tmp_path)
    section = MeasurementStage(Settings(output_dir=tmp_path), FakeDepth(depth)).run(ADDRESS)
    assert "limits" not in section


def test_a_sign_below_the_minimum_is_reported(tmp_path):
    rule = Limit("Sign height", "stop_sign_bottom_height", 100.0, "ft", 0.2, "test")
    _, section = run_with(tmp_path, [rule])

    exceeded = section["limits"]["exceeded"]
    assert len(exceeded) == 1
    assert exceeded[0]["measure"] == "stop_sign_bottom_height" and exceeded[0]["kind"] == "min" and exceeded[0]["certain"] is True
    assert exceeded[0]["value"] == pytest.approx(SIGN_BOTTOM / 0.3048, abs=0.3) and exceeded[0]["limit"] == 100.0


def test_within_the_margin_nothing_is_reported(tmp_path):
    rule = Limit("Sign height", "stop_sign_bottom_height", 100.0, "ft", 200.0, "test")  # a margin larger than the gap
    _, section = run_with(tmp_path, [rule])

    assert section["limits"]["checks"] and section["limits"]["exceeded"] == []


def test_a_rule_for_another_measure_is_not_checked(tmp_path):
    _, section = run_with(tmp_path, [Limit("Pole", "pole_height", 1.0, "m", 0.0)])
    assert section["limits"]["exceeded"] == []


def test_the_stop_sign_height_includes_the_sign(tmp_path):
    _, section = run_with(tmp_path, [])
    sign = section["stop_sign_height"]
    assert sign["value_m"] == pytest.approx(SIGN_BOTTOM, abs=0.05)
    assert sign["total_height_m"] == pytest.approx(sign["value_m"] + 0.762, abs=0.001) and sign["sign_face_m"] == 0.762


def test_the_sign_label_shows_the_total_and_the_bottom():
    block = {"source": "measured", "value_m": 2.0, "uncertainty_m": 0.05, "confidence": 0.8}
    value, detail, _ = sign_label(block, 0.762)
    assert value == "276 cm" and "to sign bottom 200 cm" in detail
    value, detail, _ = sign_label({"source": "default_max", "candidate_value_m": 1.5, "confidence": 0.2}, 0.762)
    assert value == "~226 cm" and "~150 cm" in detail
    assert sign_label({"source": "default_max", "candidate_value_m": None, "confidence": 0.0}, 0.762) is None


def test_an_exceeded_limit_turns_the_label_red_and_shows_both_numbers():
    rule = Limit("Driveway", "driveway_running_slope", 10.0, "%", 0.5)
    label = ("~6.9°", "confidence 0.40/1", (1, 2, 3))
    block = {"confidence": 0.4}

    over = evaluate([rule], "driveway_running_slope", 12.0, "x", certain=True)
    value, detail, colour = limit_label(label, over, block)
    assert (value, colour) == ("6.8°", RED) and "12.0 % · limit exceeded: max 10.0 %" in detail and "confidence 0.40/1" in detail

    maybe = evaluate([rule], "driveway_running_slope", 12.0, "x", certain=False)
    assert "may exceed the limit" in limit_label(label, maybe, block)[1]

    inside = evaluate([rule], "driveway_running_slope", 10.4, "x")
    assert limit_label(label, inside, block) == label


def test_the_measured_image_is_still_written_with_limits(tmp_path):
    path, _ = run_with(tmp_path, [Limit("Sign height", "stop_sign_bottom_height", 100.0, "ft", 0.2)])
    assert (path.parent / "measured" / "view.jpg").exists()
    assert json.loads(path.read_text())["measurements"]["limits"]["exceeded"]


def test_a_pole_with_a_sign_on_top_measures_the_sign_too(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic, "SIGN_DISTANCE", 5.0)  # closer, so the pole is tall enough in pixels to count
    path, depth = write_address(tmp_path)
    record = json.loads(path.read_text())
    for inst in record["detections"]["per_image"][0]["instances"]:
        inst.pop("pole", None)  # the detector did not link the sign to its pole
    path.write_text(json.dumps(record))
    section = MeasurementStage(Settings(output_dir=tmp_path), FakeDepth(depth)).run(ADDRESS)

    pole = section["poles"][0]
    assert pole["height_m"] == pytest.approx(SIGN_BOTTOM, abs=0.1)
    sign = pole["sign"]
    assert sign["sign_height_m"] == pytest.approx(0.762, abs=0.1) and sign["sign_height_source"] == "measured"
    assert sign["total_height_m"] == pytest.approx(SIGN_BOTTOM + 0.762, abs=0.1)
    assert "sign_line" not in pole


def test_a_driveway_width_outside_the_limits_is_flagged_and_keeps_the_angle(tmp_path):
    from measure.stage import MeasurementStage as Stage

    stage = Stage(Settings(output_dir=tmp_path), None)
    stage._limits = [Limit("min", "driveway_min_width", 10.0, "ft", 0.5), Limit("max", "driveway_max_width", 20.0, "ft", 0.5)]
    block = {"source": "measured", "confidence": 0.8}
    narrow = stage._width_checks({**block, "width_m": 2.0}, "x")
    assert [c.exceeded for c in narrow] == [False, True]  # 6.6 ft is under 10 ft
    assert not any(c.exceeded for c in stage._width_checks({**block, "width_m": 4.0}, "x"))  # 13 ft is fine
    cut = stage._width_checks({**block, "width_m": 2.0, "width_cut_off": True}, "x")
    assert cut[1].exceeded and cut[1].certain is False  # cut off by the frame: it can only be wider, so "too narrow" is uncertain
    value, detail, colour = limit_label(("4.3°", "x", (1, 2, 3)), narrow, block)
    assert value == "4.3°" and colour == RED and "width 6.6 ft" in detail and "below the limit: min 10.0 ft" in detail
