"""Checks measure.multiview.measure_ramp (the code the pipeline runs) against synthetic ramps of known slope."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "locationlabel"))
from measure.config import load_settings  # noqa: E402
from measure.fusion import ramp_block  # noqa: E402
from measure.multiview import RampView, VggtReconstructor, measure_ramp  # noqa: E402

import synth  # noqa: E402

settings = load_settings()
reconstructor = VggtReconstructor(settings.vggt_model, "cuda")
W, H, F = 1036, 728, 1040

for slope, distance, spacing in [(0.04, 8.0, 3.0), (0.083, 8.0, 3.0), (0.12, 8.0, 3.0), (0.083, 8.0, 8.0), (0.083, 12.0, 4.0)]:
    scene = synth.Scene(slope=slope, seed=3)
    views = []
    for i, position in enumerate(synth.street_view_cameras(3, spacing, distance)):
        image, labels, _ = synth.render(scene, position, np.array([0.0, 0.8, 0.06]), F, W, H, supersample=1)
        views.append(RampView(f"ramp_{i}.jpg", (image * 255).astype(np.uint8), labels == synth.RAMP, labels == synth.ROAD, {"instance_id": f"ramp_{i}", "curb_line": []}, 0.8))
    result = measure_ramp(views, reconstructor, 2.5, settings)
    block = ramp_block(True, result.running, result.cross, settings)
    truth = 100 * slope
    print(
        f"true {truth:4.1f}% ({np.degrees(np.arctan(slope)):.1f} deg), {distance:.0f} m, spacing {spacing:.0f} m -> "
        f"running {block['running_slope_pct']:.1f}%  cross {block['cross_slope_pct']:.1f}%  {block['source']}  confidence {block['confidence']:.2f}  "
        f"candidate {block.get('candidate_running_slope_pct')}  reason: {block['reason']}"
    )
