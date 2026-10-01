import json
import shutil
from pathlib import Path

import pytest

from common.cache import JsonCache
from common.record import record_path
from label_objs.config import WEIGHTS_DIR, Settings
from label_objs.detect import DetectionStage
from label_objs.rampnet import RampNetDetector

pytestmark = pytest.mark.slow


def test_real_models_on_sample_street_image(tmp_path):
    import ultralytics

    from label_objs.models import HFSegmentationModel, YoloStopSignDetector, select_device

    address = "sample street"
    path = record_path(tmp_path, address)
    (path.parent / "images").mkdir(parents=True)
    shutil.copy(Path(ultralytics.__file__).parent / "assets" / "bus.jpg", path.parent / "images" / "bus.jpg")
    path.write_text(json.dumps({"input_address": address, "streetview": {"images": [{"filename": "bus.jpg", "intrinsics": {"fx": 900.0, "fy": 900.0, "cx": 405.0, "cy": 540.0}, "pitch": -5.0}]}}))

    settings = Settings(output_dir=tmp_path, cache_path=tmp_path / "cache.sqlite")
    device = select_device()
    stage = DetectionStage(
        settings,
        HFSegmentationModel(settings.seg_model, settings.seg_input_size, device),
        YoloStopSignDetector(WEIGHTS_DIR / settings.detector_model, device),
        JsonCache(settings.cache_path),
        RampNetDetector(device, settings.ramp_min_score),
    )

    assert stage.run(address) in ("ok", "no_detections")
    counts = json.loads(path.read_text())["detections"]["summary"]["instance_counts"]
    assert counts.get("road") and counts.get("pole")
