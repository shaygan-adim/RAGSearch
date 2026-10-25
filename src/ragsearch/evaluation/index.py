"""Exact (flat) inner-product vector index on FAISS, with optional GPU search."""

from __future__ import annotations

import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


class FlatIPIndex:
    """Flat inner-product index (== cosine similarity for normalised vectors).

    Vectors are accumulated on the CPU (cheap on RAM); the first search moves the index to
    GPU 0 when ``use_gpu`` is set and a GPU build of FAISS is installed, otherwise it silently
    stays on the CPU.
    """

    def __init__(self, dim: int, use_gpu: bool = False):
        self.dim = dim
        self.use_gpu = use_gpu
        self._cpu = faiss.IndexFlatIP(dim)
        self._search_index = None
        self._gpu_resources = None

    def __len__(self) -> int:
        return self._cpu.ntotal

    def add(self, embeddings: np.ndarray) -> None:
        self._cpu.add(_as_float32(embeddings))
        self._search_index = None  # stale

    def _ready_index(self):
        if self._search_index is None:
            self._search_index = self._cpu
            if self.use_gpu:
                try:
                    self._gpu_resources = faiss.StandardGpuResources()
                    self._search_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, self._cpu)
                except (AttributeError, RuntimeError) as exc:
                    logger.warning("FAISS GPU unavailable (%s); searching on CPU", exc)
        return self._search_index

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` (scores, positions) per query; positions are -1 where fewer than ``k`` exist."""
        return self._ready_index().search(_as_float32(queries), k)

    def close(self) -> None:
        self._search_index = None
        self._gpu_resources = None
