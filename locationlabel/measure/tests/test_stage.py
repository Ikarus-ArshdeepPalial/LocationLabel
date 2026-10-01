import csv
import json

import pytest

from measure.config import Settings
from measure.evaluate import evaluate
from measure.run import update_summary
from measure.stage import MeasurementStage
from measure.tests.synthetic import ADDRESS, SIGN_BOTTOM, FakeDepth, write_address


def run_stage(tmp_path, **options):
    path, depth = write_address(tmp_path)
    model = FakeDepth(depth)
    section = MeasurementStage(Settings(output_dir=tmp_path), model, **options).run(ADDRESS)
    return path, section, model


def test_measures_synthetic_stop_sign(tmp_path):
    path, section, _ = run_stage(tmp_path, export_pointcloud=True)

    assert section["status"] == "ok"
    sign = section["stop_sign_height"]
    assert sign["source"] == "measured"
    assert sign["value_m"] == pytest.approx(SIGN_BOTTOM, abs=0.05)
    assert set(sign["methods_agreed"]) == {"flat_ground", "reference_scale", "point_cloud"}
    assert sign["methods"]["triangulation"]["reason"] == "sign seen from only one panorama"
    assert section["images"][0]["depth_scale"] == pytest.approx(1.0, abs=0.01)

    assert section["curb_height"]["reason"] == "no curb detected"
    assert section["ramp"]["source"] == "default_max"

    record = json.loads(path.read_text())
    assert "detections" in record and record["measurements"]["status"] == "ok"
    assert (path.parent / "measured" / "view.jpg").exists()
    assert (path.parent / "pointclouds" / "view.ply").exists()


def test_ramp_found_by_rampnet_without_a_region_is_not_measured(tmp_path):
    path, depth = write_address(tmp_path)
    record = json.loads(path.read_text())
    point = {"instance_id": "view_curb_ramp_0", "class": "curb_ramp", "source": "rampnet", "confidence": 0.8, "point": [330.0, 400.0],
             "mask_source": "point", "bbox": [324, 394, 336, 406], "area_px": 100, "centroid": [330.0, 400.0], "mask": "masks/view_stop_sign_0.png"}
    record["detections"]["per_image"][0]["instances"].append(point)
    path.write_text(json.dumps(record))

    section = MeasurementStage(Settings(output_dir=tmp_path), FakeDepth(depth)).run(ADDRESS)

    assert section["ramp"]["source"] == "default_max"
    assert section["ramp"]["reason"] == "ramp detected, but no ramp region was segmented to measure"


def test_depth_maps_are_cached(tmp_path):
    _, _, model = run_stage(tmp_path)
    assert model.calls == 1

    second = FakeDepth(model.depth)
    MeasurementStage(Settings(output_dir=tmp_path), second).run(ADDRESS)
    assert second.calls == 0

    forced = FakeDepth(model.depth)
    MeasurementStage(Settings(output_dir=tmp_path), forced, force=True).run(ADDRESS)
    assert forced.calls == 1


def test_broken_image_does_not_fail_address(tmp_path):
    path, depth = write_address(tmp_path)
    record = json.loads(path.read_text())
    record["detections"]["per_image"].append({"image": "missing.jpg", "semantic": "masks/none.png", "instances": []})
    record["streetview"]["images"].append({**record["streetview"]["images"][0], "filename": "missing.jpg"})
    path.write_text(json.dumps(record))

    section = MeasurementStage(Settings(output_dir=tmp_path), FakeDepth(depth)).run(ADDRESS)

    assert section["status"] == "partial"
    assert section["stop_sign_height"]["source"] == "measured"


def test_summary_and_evaluation(tmp_path):
    _, section, _ = run_stage(tmp_path)
    summary = tmp_path / "summary_batch.csv"
    update_summary(summary, {ADDRESS: section})
    row = next(csv.DictReader(summary.open()))
    assert row["stop_sign_height_source"] == "measured"

    truth = tmp_path / "ground_truth.csv"
    truth.write_text(f'address,stop_sign_height_m,curb_height_m,ramp_running_slope_pct,ramp_cross_slope_pct\n"{ADDRESS}",2.30,,,\n')
    report = evaluate(truth, tmp_path)

    sign = report["stop_sign_height_m"]
    assert sign["fused"]["n"] == 1
    assert sign["fused"]["mae"] == pytest.approx(abs(section["stop_sign_height"]["value_m"] - 2.30), abs=1e-4)
    assert {"flat_ground", "reference_scale", "point_cloud", "final"} <= set(sign)
    assert "curb_height_m" not in report


def test_user_images_with_estimated_metadata(tmp_path):
    path, depth = write_address(tmp_path)
    record = json.loads(path.read_text())
    meta = record["streetview"]["images"][0]
    meta.update(pano_lat=None, pano_lng=None, camera_height_m=2.5, metadata_sources={
        "focal_px": "estimated", "pitch_deg": "estimated", "heading_deg": "default", "camera_height_m": "default", "position": "unknown",
    })
    record["streetview"]["target"] = None
    path.write_text(json.dumps(record))
    baseline = run_stage(tmp_path / "google")[1]["stop_sign_height"]

    sign = MeasurementStage(Settings(output_dir=tmp_path), FakeDepth(depth)).run(ADDRESS)["stop_sign_height"]

    assert sign["methods"]["triangulation"]["reason"] == "camera position or heading unknown"
    assert sign["confidence"] < baseline["confidence"]
    assert sign["confidence_factors"]["metadata"] < 1
    # All camera parameters guessed: the methods still agree, but it is reported as the default with the reason.
    assert sign["source"] == "default_max"
    assert sign["reason"].endswith("camera metadata estimated or defaulted")
    assert sign["candidate_value_m"] == pytest.approx(SIGN_BOTTOM, abs=0.05)
