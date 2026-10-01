import json
from pathlib import Path

import cv2
import numpy as np

from common.record import record_path
from label_objs.config import Settings
from label_objs.models import Box, BoxDetector, RampPoint, RampPointDetector, Segmentation, SegmentationModel

ADDRESS = "2910 Sussex Gardens Ln, Austin, TX 78748, USA"
ID2LABEL = {0: "Road", 1: "Sidewalk", 2: "Curb", 3: "Curb Cut", 4: "Pole", 5: "Traffic Sign (Front)", 6: "Sky"}
SIGN_BOX = (30.0, 8.0, 54.0, 32.0)


def street_scene(with_ramp: bool = True) -> np.ndarray:
    """200x200 label map: sky, a sign on a pole, sidewalk, curb (with an optional curb cut) and road."""
    labels = np.full((200, 200), 6, np.int32)
    labels[100:140] = 1
    labels[140:145] = 2
    labels[145:] = 0
    labels[30:140, 40:44] = 4
    labels[10:30, 32:52] = 5
    if with_ramp:
        labels[130:145, 80:120] = 3
    return labels


class FakeSegmenter(SegmentationModel):
    name = "fakeseg"
    version = "fakeseg@1"
    id2label = ID2LABEL

    def __init__(self, with_ramp: bool = True, ramp_confidence: float = 0.9) -> None:
        self.with_ramp = with_ramp
        self.ramp_confidence = ramp_confidence
        self.calls = 0

    def predict(self, images: list[np.ndarray]) -> list[Segmentation]:
        self.calls += 1
        labels = street_scene(self.with_ramp)
        confidence = np.where(labels == 3, self.ramp_confidence, 0.9).astype(np.float32)
        return [Segmentation(labels, confidence) for _ in images]


class FakeDetector(BoxDetector):
    def __init__(self, name: str, boxes: list[Box]) -> None:
        self.name = name
        self.version = f"{name}@1"
        self.boxes = boxes
        self.calls = 0

    def detect(self, images: list[np.ndarray]) -> list[list[Box]]:
        self.calls += 1
        return [list(self.boxes) for _ in images]


class FakeRampDetector(RampPointDetector):
    name = "fakerampnet"
    version = "fakerampnet@1"

    def __init__(self, points: list[RampPoint]) -> None:
        self.points = points
        self.calls = []

    def detect(self, rgb: np.ndarray, fx: float, cx: float, cy: float, pitch_deg: float) -> list[RampPoint]:
        self.calls.append((fx, cx, cy, pitch_deg))
        return list(self.points)


# The synthetic scene's curb cut is rows 130-144 and columns 80-119, so this point sits on it.
RAMP_ON_CURB_CUT = RampPoint(100, 137, 0.9)


def stop_sign_detector() -> FakeDetector:
    return FakeDetector("fakeyolo", [Box("stop_sign", 0.9, SIGN_BOX), Box("stop_sign", 0.2, (150, 10, 170, 30))])


def make_settings(tmp_path: Path) -> Settings:
    return Settings(output_dir=tmp_path / "output", cache_path=tmp_path / "cache.sqlite", batch_size=2)


def write_address(tmp_path: Path, filenames: tuple[str, ...] = ("panoA_h10_p0.jpg", "panoA_h45_p0.jpg")) -> Path:
    path = record_path(tmp_path / "output", ADDRESS)
    (path.parent / "images").mkdir(parents=True)
    for name in filenames:
        cv2.imwrite(str(path.parent / "images" / name), np.full((200, 200, 3), 128, np.uint8))
    camera = {"intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 100.0, "cy": 100.0}, "pitch": 0.0, "width": 200, "height": 200}
    record = {"input_address": ADDRESS, "lat": 30.15, "streetview": {"images": [{"filename": n, **camera} for n in filenames]}}
    path.write_text(json.dumps(record))
    return path
