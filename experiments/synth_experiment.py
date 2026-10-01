"""Step 2: how accurately can VGGT + a plane fit recover a known ramp slope from Street-View-like camera positions?

Each run renders one ramp of known slope from several positions along the road, reconstructs the views with VGGT,
fits planes to the ramp and road points (using perfect masks, so this measures the geometry, not the detector), and
reports the slope error in degrees. `zoom` narrows the field of view around the ramp, like cropping a high-res photo.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "vggt_src"))
sys.path.insert(0, str(HERE.parent / "locationlabel"))
from measure.planes import fit_plane  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

import synth  # noqa: E402

WIDTH, HEIGHT, BASE_FOCAL = 518, 364, 520
TARGET = np.array([0.0, 0.8, 0.06])


def tilt_deg(normal: np.ndarray, up: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(normal @ up), 0, 1))))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1))))


def plane_of(points: np.ndarray, scale: float):
    if len(points) < 40:
        return None
    if len(points) > 20000:
        points = points[np.random.default_rng(0).choice(len(points), 20000, replace=False)]
    return fit_plane(points, 0.004 * scale, 300)


def run_case(model, slope: float, distance: float, spacing: float, zoom: float, views: int, seed: int) -> dict:
    scene = synth.Scene(slope=slope, seed=seed)
    images, labels, rotations = [], [], []
    for position in synth.street_view_cameras(views, spacing, distance):
        image, label, rotation = synth.render(scene, position, TARGET, BASE_FOCAL * zoom, WIDTH, HEIGHT)
        images.append(image)
        labels.append(label)
        rotations.append(rotation)

    batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(batch)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(out["pose_enc"], batch.shape[-2:])
    extrinsic, intrinsic = extrinsic[0].float().cpu().numpy(), intrinsic[0].float().cpu().numpy()
    depth = out["depth"][0].float().cpu().numpy()
    confidence = out["depth_conf"][0].float().cpu().numpy()
    points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)

    # Gravity: the true camera-frame "up" of each view, carried into VGGT's frame by its estimated rotation.
    ups = [extrinsic[i][:, :3].T @ (rotations[i].T @ np.array([0.0, 0.0, 1.0])) for i in range(views)]
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up)

    def gather(label_id: int, indices) -> np.ndarray:
        chosen = []
        for i in indices:
            mask = cv2.erode((labels[i] == label_id).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            mask &= confidence[i] > np.percentile(confidence[i], 30)
            chosen.append(points[i][mask])
        return np.concatenate(chosen) if chosen else np.zeros((0, 3))

    scale = float(np.median(np.linalg.norm(points[0].reshape(-1, 3), axis=1)))
    truth = float(np.degrees(np.arctan(slope)))
    ramp_all, road_all = plane_of(gather(synth.RAMP, range(views)), scale), plane_of(gather(synth.ROAD, range(views)), scale)
    result = {"slope_pct": round(100 * slope, 1), "truth_deg": round(truth, 2), "gravity_spread_deg": round(float(max(angle_between(u / np.linalg.norm(u), up) for u in ups)), 2)}
    if ramp_all is None or road_all is None:
        return {**result, "failed": True}

    per_view = []
    for i in range(views):
        plane = plane_of(gather(synth.RAMP, [i]), scale)
        if plane is not None:
            per_view.append(tilt_deg(plane.normal, up))
    result.update(
        failed=False,
        ramp_tilt_error=round(tilt_deg(ramp_all.normal, up) - truth, 2),
        relative_error=round(angle_between(ramp_all.normal, road_all.normal) - truth, 2),
        road_tilt=round(tilt_deg(road_all.normal, up), 2),
        per_view_spread=round(float(np.std(per_view)), 2) if len(per_view) > 1 else None,
        ramp_points=int(ramp_all.n_points),
    )
    return result


def main() -> None:
    model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
    slopes, seeds = (0.04, 0.083, 0.12), (0, 1)
    grid = list(itertools.product((6.0, 10.0), (3.0, 8.0), (1.0, 2.0, 4.0)))  # distance, spacing, zoom
    rows = []
    started = time.perf_counter()
    for distance, spacing, zoom in grid:
        for slope, seed in itertools.product(slopes, seeds):
            row = {"distance": distance, "spacing": spacing, "zoom": zoom, **run_case(model, slope, distance, spacing, zoom, 3, seed)}
            rows.append(row)
        subset = [r for r in rows if (r["distance"], r["spacing"], r["zoom"]) == (distance, spacing, zoom)]
        good = [r for r in subset if not r["failed"]]
        line = f"dist {distance:>4} m  spacing {spacing:>3} m  zoom {zoom:>3}:  failed {len(subset) - len(good)}/{len(subset)}"
        if good:
            line += f"   |ramp tilt err| {np.mean([abs(r['ramp_tilt_error']) for r in good]):.2f} deg   |relative err| {np.mean([abs(r['relative_error']) for r in good]):.2f} deg   road tilt {np.mean([r['road_tilt'] for r in good]):.2f} deg"
        print(line, flush=True)
    (HERE / "synth_results.json").write_text(json.dumps(rows, indent=1))
    print(f"done in {time.perf_counter() - started:.0f}s; details in synth_results.json")


if __name__ == "__main__":
    main()
