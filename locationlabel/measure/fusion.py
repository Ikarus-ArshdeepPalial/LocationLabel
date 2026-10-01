from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings
from .estimators import Estimate

M_TO_FT = 3.28084
M_TO_IN = 39.3701

FACTOR_REASONS = {
    "n_estimates": "only one estimate",
    "agreement": "estimates disagree",
    "distance": "object far from the camera",
    "mask_quality": "low detection/mask quality",
    "depth_scale": "depth scale correction far from 1",
    "method_diversity": "only one method",
    "metadata": "camera metadata estimated or defaulted",
}


@dataclass
class Fused:
    value: float
    spread: float
    uncertainty: float
    kept: list[Estimate]
    rejected: list[Estimate]
    confidence: float
    factors: dict[str, float]


def mad_inliers(values: np.ndarray, k: float, floor: float) -> np.ndarray:
    """MAD outlier test; the scale never drops below floor, so near-identical estimates can't reject each other."""
    if len(values) < 3:
        return np.ones(len(values), bool)
    median = np.median(values)
    scale = max(1.4826 * np.median(np.abs(values - median)), floor, 1e-6)
    return np.abs(values - median) <= k * scale


def confidence_factors(kept: list[Estimate], spread: float, tolerance: float, s: Settings) -> dict[str, float]:
    distances = [e.distance_m for e in kept if e.distance_m is not None]
    distance = float(np.median(distances)) if distances else s.far_distance_m
    ramp = (distance - s.near_distance_m) / (s.far_distance_m - s.near_distance_m)

    def depth_factor(e: Estimate) -> float:
        if e.method != "point_cloud":
            return 1.0
        return 0.7 if e.depth_scale is None else float(np.clip(1 - abs(e.depth_scale - 1), 0.5, 1))

    return {
        "n_estimates": 1 - 0.4 ** len(kept),
        "agreement": 1 / (1 + (spread / tolerance) ** 2),
        "distance": float(np.clip(1 - 0.8 * ramp, 0.2, 1)),
        "mask_quality": float(np.mean([e.quality for e in kept])),
        "depth_scale": float(np.mean([depth_factor(e) for e in kept])),
        "method_diversity": 1.0 if len({e.method for e in kept}) > 1 else 0.85,
        "metadata": float(np.mean([e.metadata_factor for e in kept])),
    }


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median where each value counts by its weight; equals the plain median when weights are equal."""
    order = np.argsort(values)
    values, cumulative = values[order], np.cumsum(weights[order])
    half = cumulative[-1] / 2
    i = int(np.searchsorted(cumulative, half))
    if np.isclose(cumulative[i], half) and i + 1 < len(values):
        return float((values[i] + values[i + 1]) / 2)
    return float(values[i])


def fuse(estimates: list[Estimate], plausible: tuple[float, float], tolerance: float, s: Settings) -> tuple[Fused | None, str | None]:
    applicable = [e for e in estimates if e.applicable]
    if not applicable:
        reasons = Counter(e.reason for e in estimates if e.reason)
        return None, reasons.most_common(1)[0][0] if reasons else "no estimates"

    low, high = plausible
    in_range = [e for e in applicable if low <= e.value <= high]
    if not in_range:
        return None, f"all {len(applicable)} estimates outside the plausible range {low}-{high}"

    values = np.array([e.value for e in in_range])
    inliers = mad_inliers(values, s.mad_k, float(np.median([e.uncertainty for e in in_range])))
    kept = [e for e, ok in zip(in_range, inliers) if ok]
    rejected = [e for e, ok in zip(in_range, inliers) if not ok]
    kept_values = values[inliers]

    # Estimates built on estimated/default camera metadata count less, which favours the reference-scale method.
    value = weighted_median(kept_values, np.array([e.metadata_factor for e in kept]))
    if len(kept) >= 3:
        spread = 1.4826 * float(np.median(np.abs(kept_values - value)))
    elif len(kept) == 2:
        spread = float(abs(kept_values[1] - kept_values[0]) / 2)
    else:
        spread = kept[0].uncertainty
    uncertainty = float(max(spread, np.median([e.uncertainty for e in kept]) / np.sqrt(len(kept))))
    factors = confidence_factors(kept, uncertainty, tolerance, s)
    return Fused(value, spread, uncertainty, kept, rejected, float(np.prod(list(factors.values()))), factors), None


def low_confidence_reason(fused: Fused, s: Settings) -> str:
    weakest = min(fused.factors, key=fused.factors.get)
    return f"confidence {fused.confidence:.2f} below {s.confidence_threshold}: {FACTOR_REASONS[weakest]}"


def method_summary(estimates: list[Estimate], kept: list[Estimate], key: str, plausible: tuple[float, float]) -> dict[str, Any]:
    """Per-method median of its plausible estimates, so methods can be compared and evaluated separately."""
    summary = {}
    for method in dict.fromkeys(e.method for e in estimates):
        own = [e for e in estimates if e.method == method]
        values = [e.value for e in own if e.applicable and plausible[0] <= e.value <= plausible[1]]
        if values:
            summary[method] = {key: round(float(np.median(values)), 4), "n": len(values), "n_kept": sum(e in kept for e in own)}
        else:
            reasons = Counter(e.reason or "outside plausible range" for e in own)
            summary[method] = {key: None, "reason": reasons.most_common(1)[0][0]}
    return summary


def scalar_block(kind: str, detected: bool, estimates: list[Estimate], s: Settings) -> dict[str, Any]:
    """Result for stop_sign_height or curb_height, falling back to the configured default max with a reason."""
    is_sign = kind == "stop_sign_height"
    unit_key, factor = ("value_ft", M_TO_FT) if is_sign else ("value_in", M_TO_IN)
    default = s.limits.stop_sign_default_max_m if is_sign else s.limits.curb_default_max_m
    plausible = s.plausible.stop_sign_height_m if is_sign else s.plausible.curb_height_m

    fused, reason = fuse(estimates, plausible, s.agreement_tolerance[kind], s) if detected else (None, None)
    if not detected:
        reason = "no stop sign detected" if is_sign else "no curb detected"
    elif fused and fused.confidence < s.confidence_threshold:
        reason = low_confidence_reason(fused, s)

    measured = reason is None
    value = fused.value if measured else default
    block: dict[str, Any] = {
        "value_m": round(value, 4),
        unit_key: round(value * factor, 2),
        "source": "measured" if measured else "default_max",
        "confidence": round(fused.confidence, 3) if fused else 0.0,
        "uncertainty_m": round(fused.uncertainty, 4) if fused else None,
        "spread_m": round(fused.spread, 4) if fused else None,
        "n_estimates": len(fused.kept) if fused else 0,
        "n_outliers_rejected": len(fused.rejected) if fused else 0,
        "methods": method_summary(estimates, fused.kept if fused else [], "value_m", plausible),
        "methods_agreed": sorted({e.method for e in fused.kept}) if fused else [],
        "confidence_factors": {k: round(v, 3) for k, v in fused.factors.items()} if fused else None,
        "candidate_value_m": round(fused.value, 4) if fused and not measured else None,
        "reason": reason,
    }
    if is_sign and measured:
        minimum = s.limits.stop_sign_min_height_m
        block["below_mutcd_minimum"] = value < minimum
        block["below_minimum_beyond_uncertainty"] = minimum - value > fused.uncertainty
    return block


def ramp_block(detected: bool, running: list[Estimate], cross: list[Estimate], s: Settings) -> dict[str, Any]:
    tolerance, plausible, limits = s.agreement_tolerance["ramp_slope"], s.plausible.ramp_slope_pct, s.limits
    run_fused, run_reason = fuse(running, plausible, tolerance, s) if detected else (None, "no curb ramp detected")
    cross_fused, cross_reason = fuse(cross, plausible, tolerance, s) if detected else (None, "no curb ramp detected")

    reason = run_reason or cross_reason
    confidence = min(run_fused.confidence, cross_fused.confidence) if run_fused and cross_fused else 0.0
    if reason is None and confidence < s.confidence_threshold:
        weaker = run_fused if run_fused.confidence <= cross_fused.confidence else cross_fused
        reason = low_confidence_reason(weaker, s)

    measured = reason is None
    running_pct = run_fused.value if measured else limits.ramp_running_slope_pct
    cross_pct = cross_fused.value if measured else limits.ramp_cross_slope_pct
    block: dict[str, Any] = {
        "running_slope_pct": round(running_pct, 2),
        "running_slope_deg": round(float(np.degrees(np.arctan(running_pct / 100))), 2),
        "cross_slope_pct": round(cross_pct, 2),
        "cross_slope_deg": round(float(np.degrees(np.arctan(cross_pct / 100))), 2),
        "source": "measured" if measured else "default_max",
        "confidence": round(confidence, 3),
        "uncertainty_pct": {"running": round(run_fused.uncertainty, 2), "cross": round(cross_fused.uncertainty, 2)} if run_fused and cross_fused else None,
        "n_estimates": len(run_fused.kept) if run_fused else 0,
        "methods": {
            method: {**values, "cross_slope_pct": method_summary(cross, [], "cross_slope_pct", plausible).get(method, {}).get("cross_slope_pct")}
            for method, values in method_summary(running, run_fused.kept if run_fused else [], "running_slope_pct", plausible).items()
        },
        "candidate_running_slope_pct": round(run_fused.value, 2) if run_fused and cross_fused and not measured else None,
        "candidate_cross_slope_pct": round(cross_fused.value, 2) if run_fused and cross_fused and not measured else None,
        "reason": reason,
        "exceeds_limit": False,
    }
    if measured:
        exceedance = {
            "running": (running_pct - limits.ramp_running_slope_pct, run_fused.uncertainty),
            "cross": (cross_pct - limits.ramp_cross_slope_pct, cross_fused.uncertainty),
        }
        block["exceeds_limit"] = any(excess > 0 for excess, _ in exceedance.values())
        block["exceedance"] = {
            name: {"exceeds": excess > 0, "beyond_uncertainty": excess > uncertainty, "excess_pct": round(excess, 2)}
            for name, (excess, uncertainty) in exceedance.items()
        }
    return block
