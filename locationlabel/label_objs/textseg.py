"""Text-prompted segmentation: Grounding DINO finds boxes for a prompt, and SAM outlines each box."""

import logging
from functools import cached_property

import numpy as np

from .models import Region, RegionDetector

logger = logging.getLogger(__name__)


def lift_shadows(rgb: np.ndarray) -> np.ndarray:
    """Local contrast boost on lightness; dappled tree shadow scored a driveway 0.48 raw and 0.55 with it."""
    import cv2

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class TextPromptedSegmenter(RegionDetector):
    name = "grounding_dino+sam"

    def __init__(self, device: str, grounding_model: str, sam_model: str) -> None:
        self._device = device
        self._grounding_model = grounding_model
        self._sam_model = sam_model
        self.version = f"{grounding_model}+{sam_model}"

    @cached_property
    def _grounding(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        logger.info("Loading %s on %s", self._grounding_model, self._device)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self._grounding_model).to(self._device).eval()
        return AutoProcessor.from_pretrained(self._grounding_model), model

    @cached_property
    def _sam(self):
        from transformers import SamModel, SamProcessor

        logger.info("Loading %s on %s", self._sam_model, self._device)
        model = SamModel.from_pretrained(self._sam_model).to(self._device).eval()
        return SamProcessor.from_pretrained(self._sam_model), model

    def detect(self, rgb: np.ndarray, prompt: str, min_score: float) -> list[Region]:
        import torch

        processor, grounding = self._grounding
        inputs = processor(images=lift_shadows(rgb), text=prompt, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = grounding(**inputs)
        found = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=min_score, text_threshold=min_score, target_sizes=[rgb.shape[:2]]
        )[0]
        boxes, scores = found["boxes"].cpu().numpy(), found["scores"].cpu().tolist()
        if not len(boxes):
            return []

        sam_processor, sam = self._sam
        sam_inputs = sam_processor(images=rgb, input_boxes=[boxes.tolist()], return_tensors="pt").to(self._device)
        with torch.inference_mode():
            sam_out = sam(**sam_inputs, multimask_output=False)
        masks = sam_processor.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu()
        )[0]
        return [Region(mask[0].numpy().astype(bool), float(score)) for mask, score in zip(masks, scores)]
