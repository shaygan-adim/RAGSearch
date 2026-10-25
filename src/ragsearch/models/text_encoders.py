"""Uniform ``encode(texts, mode)`` interface over the text models we compare.

All encoders return L2-normalised float32 matrices, so inner product == cosine similarity.
``mode`` is "query" or "product"; only the dual encoder treats them differently.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from ragsearch.config import ModelConfig, TextExperiment
from ragsearch.models.dual_encoder import RetrievalSystem, load_retrieval_system

logger = logging.getLogger(__name__)


class TextEncoder(Protocol):
    def encode(self, texts: list[str], mode: str = "query") -> np.ndarray: ...


def _batches(items: list[str], size: int, desc: str):
    for i in tqdm(range(0, len(items), size), desc=desc, leave=False):
        yield items[i : i + size]


def _tokenize(tokenizer, texts: list[str], max_length: int, device: str):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)


class SentenceTransformerEncoder:
    def __init__(self, model_id: str, batch_size: int, device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self.batch_size = batch_size

    def encode(self, texts: list[str], mode: str = "query") -> np.ndarray:
        embs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32)


class PooledEncoder:
    """A plain HF encoder with CLS or attention-masked mean pooling."""

    def __init__(self, model_path: str, pooling: str, max_length: int, batch_size: int, device: str):
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.pooling, self.max_length, self.batch_size, self.device = pooling, max_length, batch_size, device

    @torch.no_grad()
    def encode(self, texts: list[str], mode: str = "query") -> np.ndarray:
        out = []
        for batch in _batches(texts, self.batch_size, "Encoding"):
            inputs = _tokenize(self.tokenizer, batch, self.max_length, self.device)
            hidden = self.model(**inputs).last_hidden_state
            if self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
        return np.vstack(out)


class DualEncoder:
    """The trained :class:`RetrievalSystem`: query tower for queries, product tower for products."""

    def __init__(self, system: RetrievalSystem, tokenizer, max_length: int, batch_size: int, device: str):
        self.system, self.tokenizer = system, tokenizer
        self.max_length, self.batch_size, self.device = max_length, batch_size, device

    @torch.no_grad()
    def encode(self, texts: list[str], mode: str = "query") -> np.ndarray:
        encode_fn = self.system.encode_query if mode == "query" else self.system.encode_product
        out = []
        for batch in _batches(texts, self.batch_size, f"Encoding ({mode})"):
            inputs = _tokenize(self.tokenizer, batch, self.max_length, self.device)
            out.append(encode_fn(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy())
        return np.vstack(out)


def build_text_encoder(spec: TextExperiment, model_cfg: ModelConfig, batch_size: int, device: str) -> TextEncoder:
    """Instantiate the encoder described by one ``text_eval.experiments`` entry."""
    max_length = spec.max_length if spec.max_length is not None else model_cfg.max_length

    if spec.type == "sentence_transformer":
        return SentenceTransformerEncoder(spec.path, batch_size, device)
    if spec.type == "hf_automodel":
        return PooledEncoder(spec.path, spec.pooling, max_length, batch_size, device)
    if spec.type == "custom_dual":
        if not spec.base_path:
            raise ValueError(f"Experiment {spec.name!r}: 'base_path' is required for type 'custom_dual'")
        tokenizer = AutoTokenizer.from_pretrained(spec.base_path)
        system = load_retrieval_system(
            spec.path, spec.base_path, model_cfg.projection_dim, model_cfg.dropout, device
        )
        return DualEncoder(system, tokenizer, max_length, batch_size, device)
    raise ValueError(f"Experiment {spec.name!r}: unknown type {spec.type!r}")
