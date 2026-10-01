"""Runs the four stages for one job and turns every failure into a message a person can act on."""

import hashlib
import json
import logging
import os
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from common.progress import set_progress_listener
from common.record import load_record, record_path, slugify
from common.settings import ROOT_DIR
from measure.limits import Limit, save_limits

from .settings_store import AppSettings

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
Report = Callable[[float, str], None]

TIPS = "If the measurements look wrong, try zooming in on the ramp or driveway, or take the photo from a different angle."
APPROXIMATE = "These are approximated values from a single photo. Use them as a rough guide, not as exact measurements."
INPUTS_DIR = ROOT_DIR / "inputs"

# What can be found, in the order shown. `slope` targets need the extra ramp photos.
TARGET_LABELS = {"ramps": "Ramps", "driveways": "Driveways", "signs": "Stop signs", "poles": "Poles", "curbs": "Curbs"}
SLOPE_TARGETS = {"ramps", "driveways"}
GENERAL_TARGETS = {"signs", "poles", "curbs"}
MIN_RAMP_PHOTOS, MAX_RAMP_PHOTOS = 3, 4  # three views are required, a fourth is optional


class JobError(Exception):
    """An error whose text is already fit to show to the user."""


@dataclass
class Inputs:
    """The photos for one analysis: ordinary ones for object detection, and views of one ramp for its slope."""

    main: list[Path] = field(default_factory=list)
    ramp: list[Path] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.main) + len(self.ramp)


@dataclass
class JobResult:
    output_folder: Path | None = None
    images: list[Path] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total: int = 0


def collect_images(paths: list[Path]) -> list[Path]:
    """Image files from dropped files and folders (folders are searched one level deep), without duplicates."""
    found: dict[Path, None] = {}
    for path in paths:
        candidates = sorted(path.iterdir()) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                found[candidate.resolve()] = None
    return list(found)


def file_fingerprint(path: Path) -> str:
    """A cheap identity for a photo, to notice the same picture being added twice."""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        digest.update(handle.read(1 << 20))
    digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


def build_input_folder(inputs: Inputs, name: str) -> Path:
    """Saves the photos in the layout the pipeline reads: image.<ext> for objects, ramp/1.<ext>... for the ramp views."""
    folder = INPUTS_DIR / slugify(name)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for index, image in enumerate(inputs.main, 1):
        shutil.copy2(image, folder / f"{'image' if index == 1 else f'image_{index}'}{image.suffix.lower()}")
    if inputs.ramp:
        (folder / "ramp").mkdir()
        for index, image in enumerate(inputs.ramp, 1):
            shutil.copy2(image, folder / "ramp" / f"{index}{image.suffix.lower()}")
    return folder


def apply_limits(settings: AppSettings) -> None:
    """Writes the limits from the Limits page where the measurement stage reads them."""
    rules = []
    for entry in settings.limits:
        try:
            rules.append(Limit(**{k: entry[k] for k in ("name", "measure", "limit", "unit", "margin", "source", "enabled") if k in entry}))
        except (TypeError, KeyError):
            continue
    path = ROOT_DIR / "limits.json"
    save_limits(path, rules)
    os.environ["LIMITS_FILE"] = str(path)


def friendly_error(exc: BaseException) -> str:
    if isinstance(exc, JobError):
        return str(exc)
    lowered = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in lowered:
        return "The computer ran out of memory while analyzing. Close other programs, or use fewer or smaller photos, and try again."
    if any(word in lowered for word in ("connection", "resolve", "timed out", "offline", "huggingface", "ssl")):
        return "Could not download the analysis models. The first run needs an internet connection (a few GB, once). Check your connection and try again."
    if isinstance(exc, PermissionError):
        return "Could not write to the output folder. Choose a different folder with the output selector."
    if isinstance(exc, OSError):
        return f"A file could not be read or written: {exc}"
    return f"Something went wrong while analyzing ({type(exc).__name__}). The details were saved to the log. Try again, or try a different photo."


def run_job(folder: Path | None, address: str, settings: AppSettings, report: Report, output_dir: Path, targets: tuple[str, ...] | None = None) -> JobResult:
    use_own = folder is not None
    if not use_own and not settings.use_street_view:
        raise JobError("Add at least one photo first.")
    if not use_own and not address.strip():
        raise JobError("Type the address to fetch Street View images for, or add your own photos.")
    if not use_own and not settings.api_key.strip():
        raise JobError("Street View is turned on but there is no API key. Add it in Settings.")

    output_dir = output_dir.expanduser().resolve()
    os.environ["OUTPUT_DIR"] = str(output_dir)
    apply_limits(settings)
    os.environ.pop("GOOGLE_MAPS_API_KEY", None)
    if settings.use_street_view and settings.api_key.strip():
        os.environ["GOOGLE_MAPS_API_KEY"] = settings.api_key.strip()

    job = folder.name if use_own else address.strip()
    stages = plan_stages(use_own, job, folder, settings, targets)
    total = sum(weight for _, weight, _ in stages)
    state = {"done": 0.0, "weight": 0.0, "label": ""}

    def listener(desc: str, n: int, size: int | None) -> None:
        if not desc.startswith("[") and size:
            report((state["done"] + min(n / size, 1.0) * state["weight"]) / total, state["label"])

    set_progress_listener(listener)
    try:
        for label, weight, action in stages:
            state.update(weight=weight, label=label)
            report(state["done"] / total, label)
            check_stage(label, action(), job, output_dir)
            state["done"] += weight
    finally:
        set_progress_listener(None)
    report(1.0, "Done")
    return summarize(job, output_dir)


def plan_stages(use_own: bool, job: str, folder: Path | None, settings: AppSettings, targets: tuple[str, ...] | None) -> list[tuple[str, float, Callable[[], Counter]]]:
    from label_objs.config import TARGETS, parse_targets
    from label_objs.run import run as detect
    from measure.run import run as measure

    chosen = parse_targets(",".join(targets)) if targets else TARGETS
    has_ramp_views = use_own and (folder / "ramp").is_dir()
    detect_stage = ("Finding what you selected (the first run downloads models)", 0.5, lambda: detect([job], False, None, chosen))
    measure_label = "Estimating measurements" + (" (the first ramp slope downloads a 4.7 GB model)" if has_ramp_views else "")
    measure_stage = (measure_label, 0.35, lambda: measure([job], False, False, None))
    if use_own:
        from user_images.run import run as import_images

        read = ("Reading your photos", 0.1, lambda: import_images([(job, folder)], settings.camera_height, settings.source_type, settings.crop_ui))
        return [read, detect_stage, measure_stage]

    from addr_To_coord.run import run as geocode
    from cood_to_sview.run import run as fetch

    return [
        ("Looking up the address", 0.04, lambda: geocode([job], False, None)),
        ("Fetching Street View images", 0.11, lambda: fetch([job], False, False, False)),
        detect_stage,
        measure_stage,
    ]


def check_stage(label: str, statuses: Counter, job: str, output_dir: Path) -> None:
    bad = [s for s in statuses if s not in {"ok", "partial", "no_detections", "cached"}]
    if not bad:
        return
    detail = ""
    try:
        record = load_record(record_path(output_dir, job))
        detail = next((sec["message"] for sec in record.values() if isinstance(sec, dict) and sec.get("status") in bad and sec.get("message")), "")
    except Exception:
        logger.debug("No record to read a failure message from", exc_info=True)
    hints = {
        "Looking up the address": "The address could not be found. Check the spelling and include the city, state and ZIP code.",
        "Fetching Street View images": "No Street View images could be fetched. Check the API key, that the Street View Static API is enabled, and the address.",
        "Reading your photos": "None of the photos could be read. Use JPG or PNG files that open normally.",
    }
    raise JobError(hints.get(label, "This step failed.") + (f" ({detail})" if detail else ""))


def slope_finding(record: dict) -> str | None:
    """A line about the ramp slope from the ramp photos, when there is a usable estimate."""
    ramp = (record.get("measurements") or {}).get("ramp") or {}
    if ramp.get("method") != "multiview":
        return None
    pct = ramp.get("running_slope_pct") if ramp.get("source") == "measured" else ramp.get("candidate_running_slope_pct")
    if pct is None:
        return None
    from math import atan, degrees

    return f"Ramp slope about {degrees(atan(pct / 100)):.1f}° (confidence {ramp.get('confidence', 0):.2f} of 1)"


def count_objects(record: dict) -> tuple[int, int, int]:
    """(curb ramps, driveways, stop signs) found. All photos in the `ramp` folder show one ramp, so it counts once,
    and those photos add nothing else: they exist only to measure that ramp."""
    group = {m["filename"]: m.get("group", "main") for m in (record.get("streetview") or {}).get("images", [])}
    entries = (record.get("detections") or {}).get("per_image", [])
    has_ramp_folder = "ramp" in group.values()

    def is_driveway(instance: dict) -> bool:
        return instance.get("kind") == "driveway"

    ramps = driveways = signs = 0
    folder_ramps = []
    for entry in entries:
        in_ramp_folder = group.get(entry["image"], "main") == "ramp"
        for instance in entry["instances"]:
            if in_ramp_folder:
                if instance["class"] == "curb_ramp" and instance.get("mask_source") in ("segmentation", "driveway"):
                    folder_ramps.append(instance)
            elif instance["class"] == "stop_sign":
                signs += 1
            elif instance["class"] == "curb_ramp" and not has_ramp_folder:
                driveways, ramps = (driveways + 1, ramps) if is_driveway(instance) else (driveways, ramps + 1)
    if folder_ramps:
        if sum(map(is_driveway, folder_ramps)) * 2 > len(folder_ramps):
            driveways += 1
        else:
            ramps += 1
    return ramps, driveways, signs


def limit_findings(record: dict) -> list[str]:
    """One line per limit that was exceeded, with the measured value and the limit."""
    from measure.limits import MEASURES, format_value, from_base, to_base

    lines, seen = [], set()
    for check in ((record.get("measurements") or {}).get("limits") or {}).get("exceeded", []):
        bound = f"limit {'max' if check['kind'] == 'max' else 'min'} {format_value(check['limit'], check['unit'])}"
        if MEASURES.get(check["measure"], ("", "", ""))[2] == "slope":  # slopes lead with degrees, the percent follows
            percent = to_base(check["value"], check["unit"])
            text = f"{check['rule']}: {format_value(from_base(percent, '°'), '°')} ({format_value(percent, '%')}, {bound})"
        else:
            text = f"{check['rule']}: {format_value(check['value'], check['unit'])} ({bound})"
        if not check.get("certain", True):
            text = "May exceed: " + text
        else:
            text = ("Limit exceeded: " if check["kind"] == "max" else "Below the limit: ") + text
        if text not in seen:
            seen.add(text)
            lines.append(text)
    return lines


def summarize(job: str, output_dir: Path) -> JobResult:
    folder = record_path(output_dir, job).parent
    record = load_record(record_path(output_dir, job))
    ramps, driveways, signs = count_objects(record)

    result = JobResult(output_folder=folder, total=ramps + driveways + signs)
    for count, noun in ((ramps, "curb ramp"), (driveways, "driveway"), (signs, "stop sign")):
        if count:
            result.findings.append(f"{count} {noun}{'' if count == 1 else 's'} found")
    slope = slope_finding(record)
    if slope:
        result.findings.append(slope)
    result.findings += limit_findings(record)

    result.images = sorted((folder / "measured").glob("*.jpg"))
    if not result.images:
        result.images = sorted((folder / "annotated").glob("*.jpg"))
        result.warnings.append("Nothing could be measured with enough confidence, so the detections are shown without numbers. " + TIPS)
    if not result.findings:
        result.warnings.append("Nothing you selected was found. " + TIPS)
    if (record.get("measurements") or {}).get("status") == "partial":
        result.warnings.append("Some photos could not be measured and were skipped.")
    (folder / "app_summary.json").write_text(json.dumps({"job": job, "findings": result.findings, "warnings": result.warnings}, indent=2), encoding="utf-8")
    return result
