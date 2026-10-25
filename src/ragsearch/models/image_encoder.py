"""Image embedding models (CLIP, ConvNeXtV2, SigLIP2, DINOv3, ...) behind one interface."""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


def _hf_token() -> str | None:
    """Token for gated models (e.g. DINOv3); ``None`` falls back to the cached `huggingface-cli login`."""
    return os.environ.get("HF_TOKEN")


class ImageEncoder:
    """Loads one vision backbone and turns PIL images into L2-normalised embeddings.

    ``model_key`` picks the loading/feature-extraction recipe: keys containing ``siglip`` use
    ``get_image_features``; ``clip`` uses only the vision tower; everything else is a generic
    ``AutoModel`` pooled via ``pooler_output`` (or the CLS token).
    """

    def __init__(self, model_key: str, model_id: str, device: str):
        self.model_key, self.model_id, self.device = model_key, model_id, device
        self.model = None
        self.processor = None

    @classmethod
    def from_model_id(cls, model_id: str, device: str) -> ImageEncoder:
        """Pick the loading recipe from the model id (``siglip`` / ``clip`` / generic)."""
        lowered = model_id.lower()
        key = "siglip" if "siglip" in lowered else "clip" if "clip" in lowered else "auto"
        return cls(key, model_id, device)

    def load(self) -> ImageEncoder:
        logger.info("Loading image model %s", self.model_id)
        token = _hf_token()
        if "siglip" in self.model_key:
            # Full model: we need its `get_image_features` projection.
            self.model = AutoModel.from_pretrained(self.model_id, token=token)
            self.processor = AutoProcessor.from_pretrained(self.model_id, token=token)
        elif "clip" in self.model_key:
            self.model = CLIPModel.from_pretrained(self.model_id, token=token).vision_model
            self.processor = CLIPProcessor.from_pretrained(self.model_id, token=token)
        else:
            self.model = AutoModel.from_pretrained(self.model_id, token=token)
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, token=token)
        self.model.to(self.device).eval()
        return self

    @torch.no_grad()
    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        if "siglip" in self.model_key:
            embeddings = self.model.get_image_features(**inputs)
            if not isinstance(embeddings, torch.Tensor):  # transformers >= 5 returns a ModelOutput
                embeddings = embeddings.pooler_output
        else:
            outputs = self.model(**inputs)
            if getattr(outputs, "pooler_output", None) is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=1).cpu().numpy().astype(np.float32)

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded %s", self.model_id)


def load_images(image_dir: str | Path, keys: Sequence[str]) -> tuple[list[Image.Image], list[int]]:
    """Open ``<image_dir>/<key>.jpg`` for each key.

    Returns the images that could be read and, for each, its position in ``keys`` -
    unreadable files are skipped, so callers must use the positions to stay aligned.
    """
    images, positions = [], []
    for pos, key in enumerate(keys):
        try:
            with Image.open(Path(image_dir) / f"{key}.jpg") as img:
                images.append(img.convert("RGB"))
            positions.append(pos)
        except (OSError, ValueError):
            logger.debug("Could not read image for %s", key)
    return images, positions
