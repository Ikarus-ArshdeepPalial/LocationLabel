"""Step 3: how well does the current single-photo pipeline recover the same known ramp slopes?

Uses the same synthetic scenes, camera positions and perfect masks as synth_experiment.py, and the pipeline's own
functions: Depth Anything metric depth, ground-plane depth calibration, and the ramp plane fit. Camera "noise" mimics
the real pipeline, where pitch, camera height and focal length are estimated rather than known.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "locationlabel"))
from label_objs.postprocess import curb_line  # noqa: E402
from measure.config import load_settings  # noqa: E402
from measure.depth import HFDepthModel, depth_scale  # noqa: E402
from measure.estimators import ramp_point_cloud  # noqa: E402
from measure.geometry import Camera  # noqa: E402

import synth  # noqa: E402

WIDTH, HEIGHT, BASE_FOCAL = 1036, 728, 1040
TARGET = np.array([0.0, 0.8, 0.06])
HEIGHT_M = 2.5


def camera_for(position: np.ndarray, rotation: np.ndarray, focal: float, rng: np.random.Generator | None) -> Camera:
    forward = rotation[:, 2]
    heading = float(np.degrees(np.arctan2(forward[0], forward[1])))
    pitch = float(np.degrees(np.arcsin(forward[2])))
    height, f = position[2], focal
    if rng is not None:  # what estimated metadata looks like: pitch about 1 deg, height 0.2 m, focal 5%
        pitch += rng.normal(0, 1.0)
        height += rng.normal(0, 0.2)
        f *= 1 + rng.normal(0, 0.05)
    return Camera(f, f, WIDTH / 2, HEIGHT / 2, heading, pitch, height, WIDTH, HEIGHT)


def measure_one(depth_model, settings, scene: synth.Scene, position: np.ndarray, zoom: float, noisy: bool, rng: np.random.Generator) -> float | None:
    focal = BASE_FOCAL * zoom
    image, labels, rotation = synth.render(scene, position, TARGET, focal, WIDTH, HEIGHT, supersample=1)
    camera = camera_for(position, rotation, focal, rng if noisy else None)
    rgb = (image * 255).astype(np.uint8)
    depth = depth_model.predict(rgb)
    road, ramp = labels == synth.ROAD, labels == synth.RAMP
    scale = depth_scale(depth, camera, road, settings.depth_ground_max_distance_m)
    if scale is not None:
        depth = depth * scale
    line = curb_line(ramp, road)
    running, _ = ramp_point_cloud(camera, depth, ramp, line, settings)
    return None if not running.applicable else float(np.degrees(np.arctan(running.value / 100)))


def main() -> None:
    import torch

    settings = load_settings()
    depth_model = HFDepthModel(settings.depth_model, "cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    rows, started = [], time.perf_counter()
    for distance, zoom, noisy in itertools.product((6.0, 10.0), (1.0, 2.0), (False, True)):
        for slope, seed in itertools.product((0.04, 0.083, 0.12), (0, 1)):
            scene = synth.Scene(slope=slope, seed=seed)
            truth = float(np.degrees(np.arctan(slope)))
            for position in synth.street_view_cameras(3, 3.0, distance):
                estimate = measure_one(depth_model, settings, scene, position, zoom, noisy, rng)
                rows.append({"distance": distance, "zoom": zoom, "noisy": noisy, "slope_pct": round(100 * slope, 1), "truth_deg": round(truth, 2), "estimate_deg": None if estimate is None else round(estimate, 2), "error": None if estimate is None else round(estimate - truth, 2)})
        subset = [r for r in rows if (r["distance"], r["zoom"], r["noisy"]) == (distance, zoom, noisy)]
        good = [r for r in subset if r["error"] is not None]
        line = f"dist {distance:>4} m  zoom {zoom:>3}  camera {'noisy' if noisy else 'exact':>5}:  no estimate {len(subset) - len(good)}/{len(subset)}"
        if good:
            line += f"   |error| {np.mean([abs(r['error']) for r in good]):.2f} deg   bias {np.mean([r['error'] for r in good]):+.2f} deg"
        print(line, flush=True)
    (HERE / "synth_baseline_results.json").write_text(json.dumps(rows, indent=1))
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
