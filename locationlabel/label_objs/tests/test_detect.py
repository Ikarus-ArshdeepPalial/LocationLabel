import json
import logging

import cv2
import pytest

from common.cache import JsonCache
from label_objs.config import CLASSES
from label_objs.detect import DetectionStage
from label_objs.models import RampPoint
from label_objs.tests.fakes import (
    ADDRESS,
    RAMP_ON_CURB_CUT,
    FakeDetector,
    FakeRampDetector,
    FakeSegmenter,
    make_settings,
    stop_sign_detector,
    write_address,
)


def make_stage(tmp_path, segmenter=None, detector=None, ramp_points=(RAMP_ON_CURB_CUT,), force=False):
    settings = make_settings(tmp_path)
    return DetectionStage(
        settings,
        segmenter or FakeSegmenter(),
        detector or stop_sign_detector(),
        JsonCache(settings.cache_path),
        FakeRampDetector(list(ramp_points)),
        force,
    )


def test_labels_images_and_extends_record(tmp_path, caplog):
    record_path = write_address(tmp_path)
    caplog.set_level(logging.INFO)

    assert make_stage(tmp_path).run(ADDRESS) == "ok"

    record = json.loads(record_path.read_text())
    assert "streetview" in record
    detections = record["detections"]
    assert detections["summary"]["stop_sign_found"] and detections["summary"]["curb_ramp_found"]
    assert detections["summary"]["num_images_with_stop_sign"] == 2

    instances = detections["per_image"][0]["instances"]
    sign = next(i for i in instances if i["class"] == "stop_sign")
    assert sign["pole"]["base_pixel"] == [42, 139]
    assert sign["pole"]["instance_id"] == "panoA_h10_p0_pole_0"
    assert sign["bottom_center"] == [42.0, 32.0]
    assert sign["traffic_sign_overlap"] > 0.5

    ramp = next(i for i in instances if i["class"] == "curb_ramp")
    assert ramp["source"] == "fakerampnet" and ramp["confidence"] == 0.9
    assert ramp["mask_source"] == "segmentation" and ramp["point"] == [100.0, 137.0]
    assert ramp["curb_line"] and all(y in (145, 146) for _, y in ramp["curb_line"])

    address_dir = record_path.parent
    assert (address_dir / ramp["mask"]).exists()
    assert (address_dir / "annotated" / "panoA_h10_p0.jpg").exists()
    assert (address_dir / "contact_sheet_annotated.jpg").exists()
    semantic = cv2.imread(str(address_dir / "masks" / "panoA_h10_p0_semantic.png"), cv2.IMREAD_UNCHANGED)
    assert semantic[135, 100] == CLASSES.index("curb_ramp")
    assert json.loads((address_dir / "masks" / "labels.json").read_text())["classes"]["0"] == "unlabeled"
    assert "Dropped stop sign" in caplog.text
    assert sum(i["class"] == "stop_sign" for i in instances) == 1


def test_cached_images_are_skipped(tmp_path):
    write_address(tmp_path)
    make_stage(tmp_path).run(ADDRESS)

    segmenter = FakeSegmenter()
    assert make_stage(tmp_path, segmenter=segmenter).run(ADDRESS) == "ok"
    assert segmenter.calls == 0

    forced = FakeSegmenter()
    make_stage(tmp_path, segmenter=forced, force=True).run(ADDRESS)
    assert forced.calls == 1


def ramps_in(record_path, image=0):
    per_image = json.loads(record_path.read_text())["detections"]["per_image"][image]
    return [i for i in per_image["instances"] if i["class"] == "curb_ramp"]


def test_ramp_point_without_a_segmented_region_is_kept_as_a_point(tmp_path):
    record_path = write_address(tmp_path, ("panoA_h10_p0.jpg",))

    make_stage(tmp_path, segmenter=FakeSegmenter(with_ramp=False)).run(ADDRESS)

    (ramp,) = ramps_in(record_path)
    assert ramp["mask_source"] == "point" and "curb_line" not in ramp
    semantic = cv2.imread(str(record_path.parent / "masks" / "panoA_h10_p0_semantic.png"), cv2.IMREAD_UNCHANGED)
    assert CLASSES.index("curb_ramp") not in semantic


def test_big_confident_segmented_region_counts_without_a_rampnet_point(tmp_path):
    record_path = write_address(tmp_path, ("panoA_h10_p0.jpg",))

    make_stage(tmp_path, ramp_points=()).run(ADDRESS)

    (ramp,) = ramps_in(record_path)
    assert ramp["source"] == "fakeseg" and ramp["mask_source"] == "segmentation" and "point" not in ramp
    assert ramp["curb_line"]


def test_weak_segmented_region_needs_a_rampnet_point(tmp_path):
    record_path = write_address(tmp_path, ("panoA_h10_p0.jpg",))
    weak = FakeSegmenter(ramp_confidence=0.40)

    make_stage(tmp_path, segmenter=weak, ramp_points=()).run(ADDRESS)
    assert ramps_in(record_path) == []

    make_stage(tmp_path, segmenter=weak, force=True).run(ADDRESS)
    (ramp,) = ramps_in(record_path)
    assert ramp["source"] == "fakerampnet" and ramp["mask_source"] == "segmentation"


def test_low_scoring_ramp_points_are_dropped(tmp_path):
    record_path = write_address(tmp_path, ("panoA_h10_p0.jpg",))

    make_stage(tmp_path, segmenter=FakeSegmenter(with_ramp=False), ramp_points=[RampPoint(100, 137, 0.10), RampPoint(60, 60, 0.9)]).run(ADDRESS)

    assert [r["point"] for r in ramps_in(record_path)] == [[60.0, 60.0]]


def test_ramp_detector_gets_the_camera_of_each_image(tmp_path):
    write_address(tmp_path, ("panoA_h10_p0.jpg",))
    stage = make_stage(tmp_path)

    stage.run(ADDRESS)

    assert stage._ramp_detector.calls == [(100.0, 100.0, 100.0, 0.0)]


def test_no_detections_status(tmp_path):
    write_address(tmp_path, ("panoA_h10_p0.jpg",))

    status = make_stage(tmp_path, segmenter=FakeSegmenter(with_ramp=False), detector=FakeDetector("fakeyolo", []), ramp_points=()).run(ADDRESS)

    assert status == "no_detections"


def test_unreadable_image_gives_partial(tmp_path):
    record_path = write_address(tmp_path)
    (record_path.parent / "images" / "panoA_h45_p0.jpg").write_bytes(b"not an image")

    assert make_stage(tmp_path).run(ADDRESS) == "partial"
    detections = json.loads(record_path.read_text())["detections"]
    assert [f["image"] for f in detections["failed_images"]] == ["panoA_h45_p0.jpg"]
