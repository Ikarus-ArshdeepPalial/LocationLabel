import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def select_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class Segmentation:
    labels: np.ndarray
    confidence: np.ndarray


@dataclass
class Box:
    label: str
    score: float
    xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None


class SegmentationModel(ABC):
    name: str
    version: str
    id2label: dict[int, str]

    @abstractmethod
    def predict(self, images: list[np.ndarray]) -> list[Segmentation]:
        """Per-pixel model class ids and the probability of that class, at each image's size."""


@dataclass
class RampPoint:
    x: float
    y: float
    score: float


class RampPointDetector(ABC):
    name: str
    version: str

    @abstractmethod
    def detect(self, rgb: np.ndarray, fx: float, cx: float, cy: float, pitch_deg: float) -> list[RampPoint]:
        """Curb ramps found in an upright perspective photo, as points in image pixels."""


@dataclass
class Region:
    mask: np.ndarray
    score: float


class RegionDetector(ABC):
    name: str
    version: str

    @abstractmethod
    def detect(self, rgb: np.ndarray, prompt: str, min_score: float) -> list[Region]:
        """Regions of the photo that match a text prompt, as boolean masks with a score each."""


class BoxDetector(ABC):
    name: str
    version: str

    @abstractmethod
    def detect(self, images: list[np.ndarray]) -> list[list[Box]]:
        """Boxes (optionally with masks) for each RGB image."""


class HFSegmentationModel(SegmentationModel):
    """Mask2Former/MaskFormer or SegFormer-style semantic segmentation from the Hugging Face hub."""

    def __init__(self, model_id: str, input_size: int, device: str) -> None:
        self.name = model_id.split("/")[-1].split("-")[0]
        self._model_id = model_id
        self._input_size = input_size
        self._device = device

    @cached_property
    def _config(self):
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(self._model_id)

    @property
    def version(self) -> str:
        return f"{self._model_id}@{self._config._commit_hash}"

    @property
    def id2label(self) -> dict[int, str]:
        return {int(k): v for k, v in self._config.id2label.items()}

    @property
    def _is_mask_classifier(self) -> bool:
        return self._config.model_type in ("mask2former", "maskformer")

    @cached_property
    def _loaded(self):
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation

        logger.info("Loading segmentation model %s on %s", self._model_id, self._device)
        model_cls = AutoModelForUniversalSegmentation if self._is_mask_classifier else AutoModelForSemanticSegmentation
        model = model_cls.from_pretrained(self._model_id).to(self._device).eval()
        return AutoImageProcessor.from_pretrained(self._model_id), model

    def predict(self, images: list[np.ndarray]) -> list[Segmentation]:
        import torch
        import torch.nn.functional as F

        processor, model = self._loaded
        size = {"height": self._input_size, "width": self._input_size}
        inputs = processor(images=images, size=size, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = model(**inputs)
            if self._is_mask_classifier:
                # Same class-probability map transformers builds in post_process_semantic_segmentation.
                classes = outputs.class_queries_logits.softmax(-1)[..., :-1]
                probs = torch.einsum("bqc,bqhw->bchw", classes, outputs.masks_queries_logits.sigmoid())
            else:
                probs = outputs.logits.softmax(1)

            results = []
            for image, image_probs in zip(images, probs):
                resized = F.interpolate(image_probs[None], size=image.shape[:2], mode="bilinear", align_corners=False)[0]
                resized = resized / resized.sum(0, keepdim=True).clamp_min(1e-6)
                confidence, labels = resized.max(0)
                results.append(Segmentation(labels.cpu().numpy().astype(np.int32), confidence.cpu().numpy()))
        return results


class YoloStopSignDetector(BoxDetector):
    name = "yolo"

    def __init__(self, weights: Path, device: str, min_score: float = 0.1) -> None:
        self._weights = weights
        self._device = device
        self._min_score = min_score
        self.version = f"{weights.name}@ultralytics-{package_version('ultralytics')}"

    @cached_property
    def _loaded(self):
        from ultralytics import YOLO

        logger.info("Loading detector %s on %s", self._weights, self._device)
        self._weights.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(self._weights))
        stop_sign_id = next(i for i, name in model.names.items() if name == "stop sign")
        return model, stop_sign_id

    def detect(self, images: list[np.ndarray]) -> list[list[Box]]:
        model, stop_sign_id = self._loaded
        # PIL inputs because ultralytics treats numpy arrays as BGR.
        results = model.predict(
            [Image.fromarray(image) for image in images],
            classes=[stop_sign_id],
            conf=self._min_score,
            device=self._device,
            verbose=False,
        )
        return [
            [Box("stop_sign", float(score), tuple(map(float, xyxy))) for xyxy, score in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
            for r in results
        ]
