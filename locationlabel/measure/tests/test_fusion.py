import pytest

from common.settings import ROOT_DIR
from measure.config import Settings
from measure.estimators import Estimate, not_applicable
from measure.fusion import fuse, ramp_block, scalar_block

SETTINGS = Settings(output_dir=ROOT_DIR / "output")


def est(method: str, value: float, uncertainty: float = 0.05, distance: float = 8.0) -> Estimate:
    return Estimate(method, value, uncertainty, distance_m=distance, quality=0.9, depth_scale=1.0)


def test_mad_keeps_estimates_within_their_uncertainty():
    estimates = [est("flat_ground", 2.20), est("point_cloud", 2.20), est("reference_scale", 2.16)]

    fused, _ = fuse(estimates, (1.0, 4.0), 0.15, SETTINGS)

    assert len(fused.kept) == 3


def test_mad_rejects_outlier():
    estimates = [est("flat_ground", 2.20), est("reference_scale", 2.25), est("point_cloud", 2.21), est("flat_ground", 3.9)]

    fused, reason = fuse(estimates, (1.0, 4.0), 0.15, SETTINGS)

    assert reason is None
    assert fused.value == pytest.approx(2.21)
    assert [e.value for e in fused.rejected] == [3.9]
    assert len(fused.kept) == 3


def test_implausible_estimates_are_discarded_not_clipped():
    fused, reason = fuse([est("flat_ground", 0.4), est("point_cloud", 5.5)], (1.0, 4.0), 0.15, SETTINGS)
    assert fused is None and "plausible range" in reason

    fused, _ = fuse([est("flat_ground", 0.4), est("point_cloud", 2.3)], (1.0, 4.0), 0.15, SETTINGS)
    assert fused.value == 2.3 and len(fused.kept) == 1


def test_measured_when_confident():
    block = scalar_block("stop_sign_height", True, [est("flat_ground", 2.2), est("reference_scale", 2.24), est("point_cloud", 2.21)], SETTINGS)

    assert block["source"] == "measured"
    assert block["value_m"] == pytest.approx(2.21)
    assert block["value_ft"] == pytest.approx(2.21 * 3.28084, abs=0.01)
    assert block["confidence"] >= SETTINGS.confidence_threshold
    assert block["methods"]["reference_scale"]["value_m"] == 2.24
    assert block["below_mutcd_minimum"] is False


def test_default_max_when_not_detected():
    block = scalar_block("stop_sign_height", False, [], SETTINGS)

    assert block["source"] == "default_max"
    assert block["value_m"] == SETTINGS.limits.stop_sign_default_max_m
    assert block["reason"] == "no stop sign detected"


def test_default_max_keeps_estimator_reason():
    estimates = [not_applicable(m, "pole base occluded") for m in ("flat_ground", "reference_scale", "point_cloud")]

    block = scalar_block("stop_sign_height", True, estimates, SETTINGS)

    assert block["source"] == "default_max"
    assert block["reason"] == "pole base occluded"
    assert block["methods"]["flat_ground"] == {"value_m": None, "reason": "pole base occluded"}


def test_default_max_on_low_confidence_keeps_candidate():
    estimates = [est("flat_ground", 1.6, distance=28), est("point_cloud", 2.9, distance=28)]

    block = scalar_block("stop_sign_height", True, estimates, SETTINGS)

    assert block["source"] == "default_max"
    assert block["value_m"] == SETTINGS.limits.stop_sign_default_max_m
    assert block["candidate_value_m"] == pytest.approx(2.25)
    assert block["reason"].startswith("confidence")


def test_curb_default_uses_curb_cap():
    block = scalar_block("curb_height", False, [], SETTINGS)

    assert block["value_m"] == SETTINGS.limits.curb_default_max_m
    assert block["value_in"] == pytest.approx(9.0, abs=0.05)


def ramp_estimates(running: list[float], cross: list[float], uncertainty: float = 0.3):
    return [est("point_cloud", v, uncertainty) for v in running], [est("point_cloud", v, uncertainty) for v in cross]


def test_ramp_confident_exceedance():
    running, cross = ramp_estimates([10.1, 10.0, 9.9], [1.0, 1.1, 0.9])

    block = ramp_block(True, running, cross, SETTINGS)

    assert block["source"] == "measured"
    assert block["exceeds_limit"] is True
    assert block["exceedance"]["running"] == {"exceeds": True, "beyond_uncertainty": True, "excess_pct": pytest.approx(1.67)}
    assert block["exceedance"]["cross"]["exceeds"] is False


def test_ramp_exceedance_within_noise():
    running, cross = ramp_estimates([8.5, 8.3, 8.6], [1.0, 1.1, 0.9], uncertainty=1.0)

    block = ramp_block(True, running, cross, SETTINGS)

    assert block["source"] == "measured"
    assert block["exceeds_limit"] is True
    assert block["exceedance"]["running"]["beyond_uncertainty"] is False


def test_ramp_default_when_not_detected():
    block = ramp_block(False, [], [], SETTINGS)

    assert block["source"] == "default_max"
    assert (block["running_slope_pct"], block["cross_slope_pct"]) == (8.33, 2.08)
    assert block["reason"] == "no curb ramp detected"
    assert block["exceeds_limit"] is False


def test_weighted_median_prefers_reliable_metadata():
    flat = Estimate("flat_ground", 2.0, 0.3, distance_m=8, quality=0.9, metadata_factor=0.35)
    cloud = Estimate("point_cloud", 2.05, 0.3, distance_m=8, quality=0.9, metadata_factor=0.35)
    reference = Estimate("reference_scale", 2.3, 0.3, distance_m=8, quality=0.9, metadata_factor=0.95)

    fused, _ = fuse([flat, cloud, reference], (1.0, 4.0), 0.15, SETTINGS)

    assert fused.value == pytest.approx(2.3)
    assert fused.factors["metadata"] == pytest.approx((0.35 + 0.35 + 0.95) / 3)


def test_weighted_median_equals_median_for_equal_weights():
    from measure.fusion import weighted_median
    import numpy as np

    assert weighted_median(np.array([3.0, 1.0, 2.0]), np.ones(3)) == 2.0
    assert weighted_median(np.array([4.0, 1.0, 2.0, 3.0]), np.ones(4)) == 2.5
