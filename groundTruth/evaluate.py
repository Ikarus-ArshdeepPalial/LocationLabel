import csv
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from common.record import load_record, record_path, slugify

from .config import GROUND_TRUTH_CSV, REPORTS_DIR, RUNS_DIR

logger = logging.getLogger(__name__)

# object_type -> the Stage 3 class that stands in for "this kind of object was detected".
# sign_other has no per-code detector, so it only checks that some traffic sign was seen at all.
DETECTION_CLASS = {"curb_ramp": "curb_ramp", "stop_sign": "stop_sign", "sign_other": "traffic_sign", "sidewalk": "sidewalk"}

# ground-truth CSV column -> (measurements block, key) for attributes the pipeline currently measures.
MEASURED_ATTRIBUTES = {
    "curb_ramp": {
        "running_slope_pct": ("ramp", "running_slope_pct", "running"),
        "cross_slope_pct": ("ramp", "cross_slope_pct", "cross"),
    }
}


@dataclass
class PredictionRow:
    gt_id: str
    object_type: str
    source: str
    split: str
    is_negative: bool
    date_gap_years: float | None
    detected: bool
    detection_confidence: float | None
    measured: dict[str, float] = field(default_factory=dict)
    truth: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)
    measured_confidence: float | None = None


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def detection_metrics(rows: list[PredictionRow]) -> dict[str, Any]:
    positives = [r for r in rows if not r.is_negative]
    negatives = [r for r in rows if r.is_negative]
    confidences = [r.detection_confidence for r in positives if r.detected and r.detection_confidence is not None]
    return {
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "recall": mean_or_none([r.detected for r in positives]),
        "false_positive_rate": mean_or_none([r.detected for r in negatives]),
        "confidence_mean": mean_or_none(confidences),
        "confidence_median": median(confidences) if confidences else None,
    }


def measurement_metrics(rows: list[PredictionRow], attr: str) -> dict[str, Any]:
    """MAE/median/max/bias, and how often the error fell within our own reported uncertainty. Defaults excluded."""
    pairs = [(r.measured[attr], r.truth[attr], r.uncertainty.get(attr)) for r in rows if attr in r.measured and attr in r.truth]
    errors = [abs(p - t) for p, t, _ in pairs]
    within = [abs(p - t) <= u for p, t, u in pairs if u is not None]
    return {
        "n": len(pairs),
        "mae": mean_or_none(errors),
        "median_ae": median(errors) if errors else None,
        "max_ae": max(errors) if errors else None,
        "bias": mean_or_none([p - t for p, t, _ in pairs]),
        "within_uncertainty_rate": mean_or_none(within),
    }


def worst_errors(rows: list[PredictionRow], attr: str, n: int = 10) -> list[tuple[PredictionRow, float]]:
    scored = [(r, abs(r.measured[attr] - r.truth[attr])) for r in rows if attr in r.measured and attr in r.truth]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)[:n]


def to_float(value: str) -> float | None:
    return float(value) if value not in (None, "") else None


def load_prediction_row(gt_row: dict[str, str], record: dict[str, Any]) -> PredictionRow:
    object_type, is_negative = gt_row["object_type"], gt_row["is_negative"] in ("True", "true", "1")
    detections = (record.get("detections") or {})
    detection_class = DETECTION_CLASS[object_type]
    instances = [i for img in detections.get("per_image", []) for i in img["instances"] if i["class"] == detection_class]
    detected = bool(instances)
    confidence = max((i["confidence"] for i in instances), default=None)

    measured, truth, uncertainty, measured_confidence = {}, {}, {}, None
    for column, (block_name, key, unc_key) in MEASURED_ATTRIBUTES.get(object_type, {}).items():
        truth_value = to_float(gt_row.get(column, ""))
        if truth_value is None:
            continue
        truth[column] = truth_value
        block = (record.get("measurements") or {}).get(block_name)
        if block and block.get("source") == "measured":
            measured[column] = block[key]
            measured_confidence = block.get("confidence")
            if block.get("uncertainty_pct"):
                uncertainty[column] = block["uncertainty_pct"].get(unc_key)

    return PredictionRow(
        gt_id=gt_row["gt_id"], object_type=object_type, source=gt_row["source"], split=gt_row["split"],
        is_negative=is_negative, date_gap_years=to_float(gt_row.get("date_gap_years", "")),
        detected=detected, detection_confidence=confidence,
        measured=measured, truth=truth, uncertainty=uncertainty, measured_confidence=measured_confidence,
    )


def load_predictions(csv_path: Path = GROUND_TRUTH_CSV, runs_dir: Path = RUNS_DIR) -> list[PredictionRow]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))

    rows = []
    for gt_row in gt_rows:
        record = load_record(record_path(runs_dir, gt_row["gt_id"]))
        if not record:
            logger.warning("%s: no run output, skipped (run `groundtruth run-groundtruth` first)", gt_row["gt_id"])
            continue
        rows.append(load_prediction_row(gt_row, record))
    return rows


def group_report(rows: list[PredictionRow], group_by: str) -> dict[str, dict[str, Any]]:
    """Detection (and, for curb_ramp, measurement) metrics grouped by 'source', 'split', or 'date_gap'."""
    key = {
        "source": lambda r: r.source,
        "split": lambda r: r.split,
        "date_gap": lambda r: "gap>3y" if (r.date_gap_years or 0) > 3 else "gap<=3y",
    }[group_by]

    report = {}
    for object_type in sorted({r.object_type for r in rows}):
        by_type = [r for r in rows if r.object_type == object_type]
        report[object_type] = {}
        for group in sorted({key(r) for r in by_type}):
            subset = [r for r in by_type if key(r) == group]
            entry: dict[str, Any] = {"detection": detection_metrics(subset)}
            for attr in MEASURED_ATTRIBUTES.get(object_type, {}):
                entry[f"measurement_{attr}"] = measurement_metrics(subset, attr)
            report[object_type][group] = entry
    return report


def render_markdown(rows: list[PredictionRow]) -> str:
    lines = ["# Ground-truth evaluation", "", f"{len(rows)} rows evaluated.", ""]
    for dimension, title in (("source", "By source"), ("split", "By split"), ("date_gap", "By survey/pano date gap")):
        lines.append(f"## {title}")
        report = group_report(rows, dimension)
        for object_type, groups in report.items():
            lines.append(f"### {object_type}")
            for group, metrics in groups.items():
                d = metrics["detection"]
                recall = f"{d['recall']:.0%}" if d["recall"] is not None else "n/a"
                fpr = f"{d['false_positive_rate']:.0%}" if d["false_positive_rate"] is not None else "n/a"
                lines.append(f"- **{group}**: n={d['n_positive']}+{d['n_negative']}neg, recall={recall}, FPR={fpr}")
                for attr, m in metrics.items():
                    if attr.startswith("measurement_") and m["n"]:
                        lines.append(f"  - {attr[12:]}: n={m['n']} MAE={m['mae']:.2f} median={m['median_ae']:.2f} max={m['max_ae']:.2f} bias={m['bias']:+.2f}")
            lines.append("")
    return "\n".join(lines)


def write_predictions_csv(rows: list[PredictionRow], path: Path) -> None:
    columns = ["gt_id", "object_type", "source", "split", "is_negative", "date_gap_years", "detected", "detection_confidence"]
    attrs = sorted({attr for r in rows for attr in (r.measured.keys() | r.truth.keys())})
    for attr in attrs:
        columns += [f"{attr}_predicted", f"{attr}_truth", f"{attr}_abs_error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            row = {"gt_id": r.gt_id, "object_type": r.object_type, "source": r.source, "split": r.split,
                   "is_negative": r.is_negative, "date_gap_years": r.date_gap_years,
                   "detected": r.detected, "detection_confidence": r.detection_confidence}
            for attr in attrs:
                p, t = r.measured.get(attr), r.truth.get(attr)
                row[f"{attr}_predicted"], row[f"{attr}_truth"] = p, t
                row[f"{attr}_abs_error"] = abs(p - t) if p is not None and t is not None else None
            writer.writerow(row)


def save_worst_images(rows: list[PredictionRow], out_dir: Path, n: int = 10) -> None:
    for object_type, attrs in MEASURED_ATTRIBUTES.items():
        for attr in attrs:
            for rank, (row, error) in enumerate(worst_errors([r for r in rows if r.object_type == object_type], attr, n), 1):
                src = RUNS_DIR / slugify(row.gt_id) / "measured.jpg"
                if not src.exists():
                    continue
                dest_dir = out_dir / "worst" / attr
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest_dir / f"{rank:02d}_{row.gt_id}_error{error:.2f}.jpg")


def run_evaluate(csv_path: Path = GROUND_TRUTH_CSV, runs_dir: Path = RUNS_DIR, out_dir: Path = REPORTS_DIR) -> Path:
    rows = load_predictions(csv_path, runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "report.md"
    report_path.write_text(render_markdown(rows), encoding="utf-8")
    write_predictions_csv(rows, out_dir / "predictions_vs_truth.csv")
    save_worst_images(rows, out_dir)

    logger.info("Evaluated %d rows, report at %s", len(rows), report_path)
    return report_path
