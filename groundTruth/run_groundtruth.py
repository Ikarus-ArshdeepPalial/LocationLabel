import csv
import logging
from collections import Counter
from pathlib import Path

from common.progress import progress
from common.record import load_record, record_path

from .config import GROUND_TRUTH_CSV, RUNS_DIR

logger = logging.getLogger(__name__)


def read_rows(csv_path: Path, split: str | None) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


def already_done(gt_id: str) -> bool:
    record = load_record(record_path(RUNS_DIR, gt_id))
    return bool(record.get("measurements"))


def run(csv_path: Path = GROUND_TRUTH_CSV, split: str | None = None, limit: int | None = None, force: bool = False, dry_run: bool = False) -> Counter[str]:
    from .pipeline_mode import build_runner, run_target  # imported here: it pulls in torch, which download/build don't need

    rows = read_rows(csv_path, split)
    if limit:
        rows = rows[:limit]
    if not force:
        pending = [r for r in rows if not already_done(r["gt_id"])]
        skipped = len(rows) - len(pending)
        if skipped:
            logger.info("Resuming: %d/%d already measured, skipping", skipped, len(rows))
        rows = pending

    if dry_run:
        runner = build_runner(dry_run=True)
        planned = 0
        for row in progress(rows, "groundtruth dry-run", unit="obj", leave=True):
            status = run_target(runner, row["gt_id"], float(row["lat"]), float(row["lng"]), dry_run=True).get("streetview")
            planned += status == "dry_run"
        logger.info("Dry run: %d/%d locations would fetch Street View images (0 image requests made)", planned, len(rows))
        return Counter({"dry_run": planned})

    runner = build_runner(force=force)
    statuses = Counter()
    for row in progress(rows, "groundtruth", unit="obj", leave=True):
        result = run_target(runner, row["gt_id"], float(row["lat"]), float(row["lng"]))
        statuses[result.get("measure") or result.get("detect") or result["streetview"]] += 1
    logger.info("run-groundtruth: %s", dict(statuses))
    return statuses
