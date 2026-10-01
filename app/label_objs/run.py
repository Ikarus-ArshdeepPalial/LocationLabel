import logging
from collections import Counter

from common.cache import JsonCache
from common.progress import track_addresses

from .config import TARGETS, WEIGHTS_DIR, load_settings
from .detect import DetectionStage
from .models import HFSegmentationModel, YoloStopSignDetector, select_device

logger = logging.getLogger(__name__)


def run(addresses: list[str], force: bool = False, model: str | None = None, targets: tuple[str, ...] = TARGETS) -> Counter[str]:
    settings = load_settings(model)
    device = select_device()
    logger.info("Using device %s; detecting %s", device, ", ".join(targets))

    # Each detector is created, and its weights downloaded, only if its target was asked for.
    segmenter = HFSegmentationModel(settings.seg_model, settings.seg_input_size, device)
    sign_detector = YoloStopSignDetector(WEIGHTS_DIR / settings.detector_model, device) if "signs" in targets else None
    ramp_detector = driveway_detector = None
    if "ramps" in targets:
        from .rampnet import RampNetDetector

        ramp_detector = RampNetDetector(device, settings.ramp_min_score)
    if "driveways" in targets:
        from .textseg import TextPromptedSegmenter

        driveway_detector = TextPromptedSegmenter(device, settings.grounding_model, settings.sam_model)

    cache = JsonCache(settings.cache_path)
    try:
        stage = DetectionStage(
            settings, segmenter, sign_detector, cache, ramp_detector, force=force, driveway_detector=driveway_detector, targets=targets
        )
        return Counter(stage.run(address) for address in track_addresses(addresses, "[3/4] detect"))
    finally:
        cache.close()
