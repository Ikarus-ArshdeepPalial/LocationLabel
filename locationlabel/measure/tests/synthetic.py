"""A synthetic street: camera 2.5 m up facing north, flat road, a stop sign on a pole 8 m ahead."""

import json
from pathlib import Path

import cv2
import numpy as np

from common.record import record_path
from label_objs.config import CLASSES
from measure.depth import DepthModel
from measure.geometry import Camera

ADDRESS = "1 Synthetic St, Testville, TX 00000"
SIZE = 640
CAMERA = Camera(fx=320, fy=320, cx=320, cy=320, heading=0, pitch=0, height=2.5)
SIGN_DISTANCE = 8.0
SIGN_BOTTOM = 2.2
SIGN_WIDTH = 0.762


def project(camera: Camera, point: np.ndarray) -> tuple[float, float]:
    x, y, z = camera.rotation.T @ (np.asarray(point, float) - camera.center)
    return camera.cx + camera.fx * x / z, camera.cy + camera.fy * y / z


def scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(semantic map, depth map, sign mask, pole mask) for the synthetic street."""
    v, u = np.mgrid[0:SIZE, 0:SIZE]
    ground = CAMERA.ground_depth(u.ravel(), v.ravel()).reshape(SIZE, SIZE)
    depth = np.where(np.isfinite(ground), ground, 80.0).astype(np.float32)
    semantic = np.where(np.isfinite(ground), CLASSES.index("road"), 0).astype(np.uint8)

    left, top = project(CAMERA, [-SIGN_WIDTH / 2, SIGN_DISTANCE, SIGN_BOTTOM + SIGN_WIDTH])
    right, bottom = project(CAMERA, [SIGN_WIDTH / 2, SIGN_DISTANCE, SIGN_BOTTOM])
    _, base = project(CAMERA, [0, SIGN_DISTANCE, 0])
    sign = (u >= round(left)) & (u <= round(right)) & (v >= round(top)) & (v <= round(bottom))
    pole = (abs(u - 320) <= 2) & (v > round(bottom)) & (v <= round(base))
    for mask, cls in ((pole, "pole"), (sign, "traffic_sign")):
        semantic[mask] = CLASSES.index(cls)
        depth[mask] = SIGN_DISTANCE
    return semantic, depth, sign, pole


class FakeDepth(DepthModel):
    name = "fakedepth"
    version = "fakedepth@1"

    def __init__(self, depth: np.ndarray) -> None:
        self.depth = depth
        self.calls = 0

    def predict(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.depth


def write_address(output_dir: Path) -> tuple[Path, np.ndarray]:
    """Record with Stage 2 and Stage 3 outputs for the synthetic street; returns (record path, depth)."""
    semantic, depth, sign, pole = scene()
    path = record_path(output_dir, ADDRESS)
    folder = path.parent
    for sub in ("images", "masks"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "images" / "view.jpg"), np.full((SIZE, SIZE, 3), 128, np.uint8))
    cv2.imwrite(str(folder / "masks" / "view_semantic.png"), semantic)
    cv2.imwrite(str(folder / "masks" / "view_stop_sign_0.png"), sign.astype(np.uint8) * 255)
    cv2.imwrite(str(folder / "masks" / "view_pole_0.png"), pole.astype(np.uint8) * 255)

    ys, xs = np.nonzero(sign)
    pole_ys, _ = np.nonzero(pole)
    meta = {
        "filename": "view.jpg", "pano_id": "PANO1", "pano_lat": 30.0, "pano_lng": -97.0, "heading": 0.0, "pitch": 0.0,
        "fov": 90, "width": SIZE, "height": SIZE, "intrinsics": {"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 320.0},
    }
    instances = [
        {"instance_id": "view_pole_0", "class": "pole", "confidence": 0.9, "mask": "masks/view_pole_0.png"},
        {
            "instance_id": "view_stop_sign_0", "class": "stop_sign", "confidence": 0.9, "mask": "masks/view_stop_sign_0.png",
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "bottom_center": [320.0, float(ys.max())],
            "pole": {"instance_id": "view_pole_0", "base_pixel": [320, int(pole_ys.max())]},
        },
    ]
    record = {
        "input_address": ADDRESS,
        "status": "ok",
        "streetview": {"images": [meta], "target": {"adjusted": {"lat": 30.0001, "lng": -97.0}}},
        "detections": {"per_image": [{"image": "view.jpg", "semantic": "masks/view_semantic.png", "instances": instances}]},
    }
    path.write_text(json.dumps(record))
    return path, depth
