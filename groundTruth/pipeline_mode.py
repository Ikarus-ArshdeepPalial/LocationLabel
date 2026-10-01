"""Runs Stages 2-4 on a lat/lng directly, aimed at the ground-truth object itself (no geocoding, no address)."""

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from common.cache import JsonCache
from common.record import record_path, update_record
from cood_to_sview.config import load_settings as load_streetview_settings
from cood_to_sview.provider import GoogleStreetViewProvider, build_session
from cood_to_sview.streetview import StreetViewStage
from label_objs.config import WEIGHTS_DIR, load_settings as load_detect_settings
from label_objs.detect import DetectionStage
from label_objs.models import HFSegmentationModel, YoloStopSignDetector, select_device
from label_objs.rampnet import RampNetDetector
from measure.config import load_settings as load_measure_settings
from measure.depth import HFDepthModel
from measure.stage import MeasurementStage

from .config import RUNS_DIR

logger = logging.getLogger(__name__)


@dataclass
class Runner:
    """Lazily-loaded stage instances for `run_target`, so the ML models load once per `run-groundtruth` run."""

    streetview: StreetViewStage
    detect: DetectionStage
    measure: MeasurementStage


def build_runner(force: bool = False, dry_run: bool = False) -> Runner:
    sv_settings = replace(load_streetview_settings(), output_dir=RUNS_DIR)
    if not sv_settings.google_api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set; it's needed to fetch Street View images")
    session = build_session(sv_settings.max_retries)
    sv_provider = GoogleStreetViewProvider(sv_settings.google_api_key, session, sv_settings.timeout)
    streetview = StreetViewStage(sv_settings, sv_provider, JsonCache(sv_settings.cache_path), session, dry_run=dry_run, force=force)

    detect_settings = replace(load_detect_settings(), output_dir=RUNS_DIR)
    device = select_device()
    detect = DetectionStage(
        detect_settings,
        HFSegmentationModel(detect_settings.seg_model, detect_settings.seg_input_size, device),
        YoloStopSignDetector(WEIGHTS_DIR / detect_settings.detector_model, device),
        JsonCache(detect_settings.cache_path),
        RampNetDetector(device, detect_settings.ramp_min_score),
        force=force,
    )

    measure_settings = replace(load_measure_settings(), output_dir=RUNS_DIR)
    measure = MeasurementStage(measure_settings, HFDepthModel(measure_settings.depth_model, device), force=force)
    return Runner(streetview, detect, measure)


def seed_record(gt_id: str, lat: float, lng: float) -> None:
    """Writes a Stage-1-shaped record so Stage 2 can run without ever calling the geocoder."""
    path = record_path(RUNS_DIR, gt_id)
    update_record(path, {
        "input_address": gt_id,
        "formatted_address": gt_id,
        "lat": lat,
        "lng": lng,
        "provider": "ground_truth",
        "match_quality": "EXACT",
        "status": "ok",
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def run_target(runner: Runner, gt_id: str, lat: float, lng: float, dry_run: bool = False) -> dict[str, str]:
    """Runs Stage 2 (or, with a dry-run runner, just resolves panos/views), then Stages 3-4 unless dry-run."""
    seed_record(gt_id, lat, lng)
    statuses = {"streetview": runner.streetview.run(gt_id)}
    if dry_run or statuses["streetview"] not in ("ok", "partial"):
        return statuses
    statuses["detect"] = runner.detect.run(gt_id)
    statuses["measure"] = runner.measure.run(gt_id)["status"]
    return statuses
