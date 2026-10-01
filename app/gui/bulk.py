"""Bulk analysis: many folders, one location each. The structure is checked first, the folders are analyzed in
chunks (so the models load once per chunk, not once per folder), and the results come out in the same layout.

    main folder/                     <- the input folder you pick
        any name/                    <- one location
            any name.png             <- the photo for objects (signs, poles, curbs)
            ramp/                    <- 3 or more photos of one ramp or driveway from different positions
                1.png 2.png 3.png
"""

import csv
import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from common.progress import set_progress_listener
from common.record import load_record, record_path
from common.settings import ROOT_DIR

from .pipeline import GENERAL_TARGETS, IMAGE_SUFFIXES, MIN_RAMP_PHOTOS, SLOPE_TARGETS, apply_limits, friendly_error, summarize
from .settings_store import AppSettings

logger = logging.getLogger(__name__)

RAMP_FOLDER = "ramp"
CHUNK = 8
STAGES = (("Reading photos", 0.1), ("Finding objects", 0.55), ("Measuring", 0.35))


@dataclass
class JobFolder:
    name: str
    path: Path
    main: list[Path] = field(default_factory=list)
    ramp: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class Scan:
    jobs: list[JobFolder]
    loose_images: int  # photos lying directly in the input folder, which belong to no location


@dataclass
class JobOutcome:
    name: str
    status: str  # "done", "failed", "skipped" or "stopped"
    message: str = ""
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def images_in(folder: Path) -> list[Path]:
    try:
        return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=lambda p: p.name.lower())
    except OSError:
        return []


def subfolders(folder: Path) -> list[Path]:
    try:
        return sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: p.name.lower())
    except OSError:
        return []


def scan_folder(root: Path, targets: tuple[str, ...]) -> Scan:
    """Checks the structure, not the names: each folder inside `root` is one location, its photos are the objects photo,
    and a folder called `ramp` holds the ramp views. What is required depends on what you want to find."""
    wants_general, wants_slope = bool(set(targets) & GENERAL_TARGETS), bool(set(targets) & SLOPE_TARGETS)
    jobs = []
    for entry in subfolders(root):
        if entry.name.startswith((".", "_")):
            continue
        job = JobFolder(entry.name, entry)
        job.main = images_in(entry)
        ramp_dir = next((p for p in subfolders(entry) if p.name.lower() == RAMP_FOLDER), None)
        job.ramp = images_in(ramp_dir) if ramp_dir else []
        if wants_general and not job.main:
            job.problems.append("no photo in the folder (needed for signs, poles or curbs)")
        if wants_slope:
            if ramp_dir is None:
                job.problems.append(f"no '{RAMP_FOLDER}' folder (needed for ramps and driveways)")
            elif len(job.ramp) < MIN_RAMP_PHOTOS:
                job.problems.append(f"only {len(job.ramp)} photo{'s' if len(job.ramp) != 1 else ''} in '{ramp_dir.name}', {MIN_RAMP_PHOTOS} are needed")
        stray = next((p.name for p in subfolders(entry) if p != ramp_dir and images_in(p)), None)
        if job.problems and stray:
            job.problems.append(f"photos were found in '{stray}': put them directly in the folder or in '{RAMP_FOLDER}'")
        if not job.main and not job.ramp and not job.problems:
            job.problems.append("the folder is empty")
        jobs.append(job)
    return Scan(jobs, len(images_in(root)))


# -- running ---------------------------------------------------------------------------------------------------


def unique_name(stem: str, used: set[str]) -> str:
    name, n = stem, 2
    while name.lower() in used:
        name, n = f"{stem}_{n}", n + 1
    used.add(name.lower())
    return name


def export_job(address: str, job: JobFolder, work_root: Path, out_root: Path, outcome: JobOutcome) -> None:
    """Copies the labeled images into the output in the same layout as the input, with a summary beside them."""
    record = load_record(record_path(work_root, address))
    work = record_path(work_root, address).parent
    out_dir = out_root / job.name
    out_dir.mkdir(parents=True, exist_ok=True)
    used: dict[Path, set[str]] = {}
    for meta in record["streetview"]["images"]:
        target = out_dir / RAMP_FOLDER if meta.get("group") == "ramp" else out_dir
        stem = unique_name(Path(meta["original_file"]).stem, used.setdefault(target, set()))
        done = Path(meta["filename"]).stem
        source = next((p for p in (work / "measured" / f"{done}.jpg", work / "annotated" / f"{done}.jpg") if p.exists()), None)
        if source:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / f"{stem}.jpg")
    (out_dir / "summary.json").write_text(json.dumps({"folder": job.name, "findings": outcome.findings, "warnings": outcome.warnings}, indent=2), encoding="utf-8")


def failure_reason(address: str, work_root: Path) -> str:
    """A plain reason from the stage that failed for this folder, or '' when none did."""
    try:
        record = load_record(record_path(work_root, address))
    except Exception:
        return "nothing was produced for this folder"
    for key, label in (("streetview", "the photos could not be read"), ("detections", "finding objects failed"), ("measurements", "measuring failed")):
        section = record.get(key) or {}
        if section.get("status") == "error":
            return f"{label}" + (f" ({section['message']})" if section.get("message") else "")
    return ""


def run_bulk(
    jobs: list[JobFolder],
    targets: tuple[str, ...],
    settings: AppSettings,
    out_root: Path,
    report: Callable[[float, str], None],
    should_stop: Callable[[], bool],
    skip_done: bool = False,
) -> list[JobOutcome]:
    from label_objs.config import parse_targets
    from label_objs.run import run as detect
    from measure.run import run as measure
    from user_images.run import run as import_images

    chosen = parse_targets(",".join(targets))
    work_root = ROOT_DIR / "bulk_work" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ["OUTPUT_DIR"] = str(work_root)
    apply_limits(settings)
    os.environ.pop("GOOGLE_MAPS_API_KEY", None)

    outcomes: list[JobOutcome] = []
    todo: list[JobFolder] = []
    for job in jobs:
        if skip_done and (out_root / job.name / "summary.json").exists():
            outcomes.append(JobOutcome(job.name, "skipped", "already has results"))
        else:
            todo.append(job)

    total, finished = len(todo), 0
    state = {"base": 0.0, "span": 0.0, "label": ""}

    def listener(desc: str, n: int, size: int | None) -> None:
        if desc.startswith("[") and size:  # the stage-level bar: one step per folder
            report(state["base"] + state["span"] * n / size, f"{state['label']} ({finished + min(n, size)} of {total} folders)")

    set_progress_listener(listener)
    try:
        for start in range(0, total, CHUNK):
            if should_stop():
                outcomes += [JobOutcome(job.name, "stopped", "stopped before it was analyzed") for job in todo[start:]]
                break
            chunk = todo[start:start + CHUNK]
            addresses = {job.name: f"{start + i:04d} {job.name}" for i, job in enumerate(chunk)}
            stages = [
                lambda: import_images([(addresses[j.name], j.path) for j in chunk], settings.camera_height, settings.source_type, settings.crop_ui),
                lambda: detect([addresses[j.name] for j in chunk], False, None, chosen),
                lambda: measure([addresses[j.name] for j in chunk], False, False, None),
            ]
            chunk_share, offset, failed_with = len(chunk) / total, 0.0, ""
            for (label, weight), stage in zip(STAGES, stages):
                state.update(base=(finished / total) + chunk_share * offset, span=chunk_share * weight, label=label)
                report(state["base"], f"{label} ({finished} of {total} folders)")
                try:
                    stage()
                except Exception as exc:
                    logger.exception("Bulk stage %s failed", label)
                    failed_with = friendly_error(exc)
                    break
                offset += weight
            for job in chunk:
                outcomes.append(finish_job(job, addresses[job.name], work_root, out_root, failed_with))
                shutil.rmtree(record_path(work_root, addresses[job.name]).parent, ignore_errors=True)
            finished += len(chunk)
            report(finished / total, f"{finished} of {total} folders done")
    finally:
        set_progress_listener(None)
        shutil.rmtree(work_root, ignore_errors=True)
        write_csv(out_root / "bulk_summary.csv", outcomes)
    return outcomes


def finish_job(job: JobFolder, address: str, work_root: Path, out_root: Path, stage_failure: str) -> JobOutcome:
    reason = stage_failure or failure_reason(address, work_root)
    if reason:
        return JobOutcome(job.name, "failed", reason)
    try:
        result = summarize(address, work_root)
        outcome = JobOutcome(job.name, "done", findings=result.findings, warnings=result.warnings)
        export_job(address, job, work_root, out_root, outcome)
        return outcome
    except Exception as exc:
        logger.exception("Could not export %s", job.name)
        return JobOutcome(job.name, "failed", friendly_error(exc))


CSV_COLUMNS = ["folder", "status", "findings", "warnings", "message"]


def write_csv(path: Path, outcomes: list[JobOutcome]) -> None:
    """One row per folder. Folders that were skipped or not reached keep the row from an earlier run, so a partial rerun
    adds to the summary instead of erasing it."""
    rows: dict[str, dict[str, str]] = {}
    try:
        if path.exists():
            with path.open(newline="", encoding="utf-8-sig") as handle:
                rows = {row["folder"]: row for row in csv.DictReader(handle) if row.get("folder")}
        for o in outcomes:
            if o.status in ("skipped", "stopped") and o.name in rows:
                continue
            rows[o.name] = {"folder": o.name, "status": o.status, "findings": "; ".join(o.findings), "warnings": "; ".join(o.warnings), "message": o.message}
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows[name] for name in sorted(rows, key=str.lower))
    except OSError:
        logger.exception("Could not write %s", path)
