"""RampNet curb-ramp detector from Project Sidewalk (MIT); it expects a 360-degree equirectangular panorama.

KeypointModel is copied from https://github.com/ProjectSidewalk/RampNet (rampnet/model.py). Weights are read from
safetensors files, so no code from the Hugging Face repositories is executed.
"""

import math
from dataclasses import dataclass
from functools import cache, cached_property

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy.ndimage import maximum_filter

from .models import RampPoint, RampPointDetector

BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
PANO_INPUT_HW, PANO_HEATMAP_HW = (2048, 4096), (512, 1024)
FILL = (124, 116, 104)  # ImageNet mean colour, so the empty part of the canvas is neutral to the network
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class KeypointModel(nn.Module):
    def __init__(self, heatmap_hw: tuple[int, int]) -> None:
        super().__init__()
        backbone = timm.create_model(BACKBONE, pretrained=False)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(backbone.num_features, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=heatmap_hw, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_extractor(image))


@dataclass
class Peak:
    x: float
    y: float
    score: float


def load_model(repo: str, filename: str, heatmap_hw: tuple[int, int], device: str) -> KeypointModel:
    model = KeypointModel(heatmap_hw)
    state = load_file(hf_hub_download(repo, filename))
    model.load_state_dict({k.removeprefix("model."): v for k, v in state.items()})
    return model.to(device).eval()


def to_tensor(rgb: np.ndarray, hw: tuple[int, int]) -> torch.Tensor:
    resized = cv2.resize(rgb, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(((resized.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))


def heatmap_peaks(heatmap: np.ndarray, min_distance: int, min_score: float) -> list[tuple[float, float, float]]:
    """Local maxima at least min_distance apart, as (x, y, score) in heatmap pixels."""
    h = np.clip(heatmap, 0, 1)
    is_peak = (h == maximum_filter(h, size=2 * min_distance + 1)) & (h >= min_score)
    ys, xs = np.nonzero(is_peak)
    return [(float(x), float(y), float(h[y, x])) for y, x in zip(ys, xs)]


class PanoRampDetector:
    """The published ramp detector; expects a 360-degree equirectangular panorama."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.model = load_model("projectsidewalk/rampnet-model", "model.safetensors", PANO_HEATMAP_HW, device)

    @torch.inference_mode()
    def detect(self, pano_rgb: np.ndarray, min_score: float = 0.2) -> list[Peak]:
        """Peaks in the panorama's own pixel coordinates."""
        h, w = pano_rgb.shape[:2]
        batch = to_tensor(pano_rgb, PANO_INPUT_HW)[None].to(self.device)
        try:
            heatmap = self.model(batch)[0, 0]
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16):
                heatmap = self.model(batch)[0, 0]
        scale_x, scale_y = w / PANO_HEATMAP_HW[1], h / PANO_HEATMAP_HW[0]
        return [Peak((x + 0.5) * scale_x, (y + 0.5) * scale_y, s) for x, y, s in heatmap_peaks(heatmap.float().cpu().numpy(), 10, min_score)]


@cache
def canvas_rays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit direction (x east, y up, z north) of every pixel of the equirectangular canvas."""
    h, w = PANO_INPUT_HW
    lon, lat = np.meshgrid(((np.arange(w) + 0.5) / w - 0.5) * 2 * np.pi, (0.5 - (np.arange(h) + 0.5) / h) * np.pi)
    lon, lat = lon.astype(np.float32), lat.astype(np.float32)
    return np.sin(lon) * np.cos(lat), np.sin(lat), np.cos(lon) * np.cos(lat)


def place_in_panorama(rgb: np.ndarray, fx: float, cx: float, cy: float, pitch_deg: float) -> np.ndarray:
    """The photo at its true place in an otherwise empty equirectangular canvas (camera facing the canvas centre)."""
    dx, dy, dz = canvas_rays()
    pitch = math.radians(pitch_deg)
    y = dy * math.cos(pitch) - dz * math.sin(pitch)
    z = dy * math.sin(pitch) + dz * math.cos(pitch)
    with np.errstate(divide="ignore", invalid="ignore"):
        map_x = np.where(z > 1e-6, cx + fx * dx / z, -1).astype(np.float32)
        map_y = np.where(z > 1e-6, cy - fx * y / z, -1).astype(np.float32)
    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=FILL)


def canvas_point_to_image(px: float, py: float, fx: float, cx: float, cy: float, pitch_deg: float, shape: tuple[int, ...]) -> tuple[float, float] | None:
    """Where a canvas pixel lands in the photo, or None if it is outside it (the inverse of place_in_panorama)."""
    h, w = PANO_INPUT_HW
    lon, lat = (px / w - 0.5) * 2 * math.pi, (0.5 - py / h) * math.pi
    dx, dy, dz = math.sin(lon) * math.cos(lat), math.sin(lat), math.cos(lon) * math.cos(lat)
    pitch = math.radians(pitch_deg)
    y, z = dy * math.cos(pitch) - dz * math.sin(pitch), dy * math.sin(pitch) + dz * math.cos(pitch)
    if z <= 1e-6:
        return None
    u, v = cx + fx * dx / z, cy - fx * y / z
    return (u, v) if 0 <= u < shape[1] and 0 <= v < shape[0] else None


class RampNetDetector(RampPointDetector):
    """RampNet on an ordinary photo: place it in an empty panorama, detect, and map the points back."""

    name = "rampnet"

    def __init__(self, device: str, min_score: float) -> None:
        self._device = device
        self._min_score = min_score
        self.version = "projectsidewalk/rampnet-model"

    @cached_property
    def _model(self) -> PanoRampDetector:
        return PanoRampDetector(self._device)

    def detect(self, rgb: np.ndarray, fx: float, cx: float, cy: float, pitch_deg: float) -> list[RampPoint]:
        canvas = place_in_panorama(rgb, fx, cx, cy, pitch_deg)
        points = []
        for peak in self._model.detect(canvas, self._min_score):
            xy = canvas_point_to_image(peak.x, peak.y, fx, cx, cy, pitch_deg, rgb.shape)
            if xy:
                points.append(RampPoint(xy[0], xy[1], peak.score))
        return points
