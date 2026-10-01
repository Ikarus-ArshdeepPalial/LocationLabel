"""Compare pipeline measurements with a ground_truth.csv, per method and fused."""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common.record import load_record, record_path

logger = logging.getLogger(__name__)

# ground-truth column -> (measurements block, value key used by the block and its per-method entries)
TARGETS = {
    "stop_sign_height_m": ("stop_sign_height", "value_m"),
    "curb_height_m": ("curb_height", "value_m"),
    "ramp_running_slope_pct": ("ramp", "running_slope_pct"),
    "ramp_cross_slope_pct": ("ramp", "cross_slope_pct"),
}


def collect_errors(ground_truth: list[dict[str, str]], output_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Signed errors (predicted - truth) per target and per method, plus 'fused' (measured only) and 'final'."""
    errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in ground_truth:
        measurements = load_record(record_path(output_dir, row["address"])).get("measurements")
        if not measurements or measurements.get("status") == "error":
            logger.warning("No measurements for %r, skipped", row["address"])
            continue
        for column, (block_name, key) in TARGETS.items():
            if not (row.get(column) or "").strip():
                continue
            truth = float(row[column])
            block = measurements[block_name]
            errors[column]["final"].append(block[key] - truth)
            if block["source"] == "measured":
                errors[column]["fused"].append(block[key] - truth)
            for method, result in block["methods"].items():
                if result.get(key) is not None:
                    errors[column][method].append(result[key] - truth)
    return errors


def summarize(errors: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, dict[str, Any]]]:
    report = {}
    for column, by_method in errors.items():
        report[column] = {}
        for method, values in by_method.items():
            signed = np.array(values)
            absolute = np.abs(signed)
            report[column][method] = {
                "n": len(values),
                "mae": round(float(absolute.mean()), 4),
                "median_ae": round(float(np.median(absolute)), 4),
                "max_ae": round(float(absolute.max()), 4),
                "bias": round(float(signed.mean()), 4),
            }
    return report


def evaluate(ground_truth_csv: Path, output_dir: Path) -> dict[str, Any]:
    with ground_truth_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    report = summarize(collect_errors(rows, output_dir))

    for column, methods in report.items():
        logger.info("%s", column)
        for method, stats in sorted(methods.items(), key=lambda item: item[1]["mae"]):
            logger.info("  %-16s n=%-3d MAE=%-8.4f median=%-8.4f max=%-8.4f bias=%+.4f",
                        method, stats["n"], stats["mae"], stats["median_ae"], stats["max_ae"], stats["bias"])

    out_path = output_dir / f"evaluation_{ground_truth_csv.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Evaluation saved to %s", out_path)
    return report
