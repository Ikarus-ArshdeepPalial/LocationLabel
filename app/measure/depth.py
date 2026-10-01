import logging
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

import numpy as np

from .geometry import Camera

logger = logging.getLogger(__name__)


class DepthModel(ABC):
    name: str
    version: str

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Metric camera z-depth in metres for each pixel of an RGB image."""


class HFDepthModel(DepthModel):
    """Metric depth models served by transformers' depth-estimation classes (e.g. Depth Anything V2 Metric)."""

    def __init__(self, model_id: str, device: str) -> None:
        self.name = model_id.split("/")[-1]
        self.version = model_id
        self._model_id = model_id
        self._device = device

    @cached_property
    def _loaded(self):
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForDepthEstimation

        config = AutoConfig.from_pretrained(self._model_id)
        if getattr(config, "depth_estimation_type", "metric") != "metric":
            raise ValueError(f"{self._model_id} predicts relative depth; a metric model is required")
        logger.info("Loading depth model %s on %s", self._model_id, self._device)
        model = AutoModelForDepthEstimation.from_pretrained(self._model_id).to(self._device).eval()
        self.version = f"{self._model_id}@{config._commit_hash}"
        return AutoImageProcessor.from_pretrained(self._model_id), model

    def predict(self, image: np.ndarray) -> np.ndarray:
        import torch

        processor, model = self._loaded
        inputs = processor(images=image, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = model(**inputs)
        depth = processor.post_process_depth_estimation(outputs, target_sizes=[image.shape[:2]])[0]["predicted_depth"]
        return depth.float().cpu().numpy()


class DepthCache:
    """Depth maps saved as float16 .npy files next to the address's other outputs."""

    def __init__(self, model: DepthModel, depth_dir: Path, force: bool) -> None:
        self._model = model
        self._dir = depth_dir
        self._force = force

    def get(self, stem: str, image: np.ndarray) -> np.ndarray:
        path = self._dir / f"{stem}.{self._model.name}.npy"
        if path.exists() and not self._force:
            return np.load(path).astype(np.float32)
        depth = self._model.predict(image)
        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(path, depth.astype(np.float16))
        return depth


def depth_scale(depth: np.ndarray, camera: Camera, ground_mask: np.ndarray, max_distance: float) -> float | None:
    """Median ratio of flat-ground depth to predicted depth over nearby ground pixels."""
    v, u = np.nonzero(ground_mask[::2, ::2])
    u, v = u * 2, v * 2
    expected = camera.ground_depth(u, v)
    predicted = depth[v, u]
    near = camera.ground_distance(u, v) < max_distance
    valid = np.isfinite(expected) & near & (predicted > 0)
    if valid.sum() < 50:
        return None
    return float(np.median(expected[valid] / predicted[valid]))
