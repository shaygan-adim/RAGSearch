"""Search backend for the demo: two FAISS indexes over the catalogue.

* text index  - every product, embedded with the dual encoder's *product* tower;
                queries use the *query* tower;
* image index - products that have an image, embedded with the image model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ragsearch.config import Config
from ragsearch.evaluation.index import FlatIPIndex
from ragsearch.models.dual_encoder import load_retrieval_system
from ragsearch.models.image_encoder import ImageEncoder, load_images
from ragsearch.models.text_encoders import DualEncoder
from ragsearch.utils import resolve_device

logger = logging.getLogger(__name__)

NO_BRAND = "No Brand"
INDEX_CHUNK = 10_000


@dataclass
class SearchHit:
    rank: int
    score: float
    persian_name: str
    category_name: str
    brand_name: str
    image_path: str | None


class SearchEngine:
    def __init__(self, products: pd.DataFrame, image_dir: str | Path, text_encoder, image_encoder, top_k: int,
                 batch_size: int, use_gpu: bool):
        self.image_dir = Path(image_dir)
        self.top_k = top_k
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder

        have_image = {p.stem for p in self.image_dir.glob("*.jpg")}
        self.text_products = products.reset_index(drop=True)
        self.image_products = self.text_products[self.text_products["random_key"].isin(have_image)].reset_index(drop=True)
        logger.info("%d products for text search, %d with an image for image search",
                    len(self.text_products), len(self.image_products))

        self.text_index = self._build_text_index(use_gpu)
        self.image_index = self._build_image_index(batch_size, use_gpu)

    @classmethod
    def from_config(cls, cfg: Config) -> SearchEngine:
        device = resolve_device()
        acfg = cfg.app

        system = load_retrieval_system(cfg.paths.checkpoint_path, acfg.text_base_model_id,
                                       cfg.model.projection_dim, cfg.model.dropout, device)
        tokenizer = AutoTokenizer.from_pretrained(acfg.text_base_model_id)
        text_encoder = DualEncoder(system, tokenizer, cfg.model.max_length, acfg.batch_size, device)

        image_encoder = ImageEncoder.from_model_id(acfg.image_model_id, device).load()
        products = pd.read_parquet(cfg.paths.products_path)
        return cls(products, cfg.paths.image_dir, text_encoder, image_encoder, acfg.top_k, acfg.batch_size,
                   acfg.use_faiss_gpu)

    def _build_text_index(self, use_gpu: bool) -> FlatIPIndex:
        texts = self.text_products["cleaned_text"].fillna("").astype(str).tolist()
        index = None
        for start in tqdm(range(0, len(texts), INDEX_CHUNK), desc="Indexing text"):
            embs = self.text_encoder.encode(texts[start : start + INDEX_CHUNK], mode="product")
            if index is None:
                index = FlatIPIndex(embs.shape[1], use_gpu)
            index.add(embs)
        if index is None:
            raise ValueError("The product table is empty; nothing to index")
        return index

    def _build_image_index(self, batch_size: int, use_gpu: bool) -> FlatIPIndex | None:
        keys = self.image_products["random_key"].tolist()
        index = None
        kept: list[int] = []
        for start in tqdm(range(0, len(keys), batch_size), desc="Indexing images"):
            images, positions = load_images(self.image_dir, keys[start : start + batch_size])
            if not images:
                continue
            embs = self.image_encoder.encode(images)
            if index is None:
                index = FlatIPIndex(embs.shape[1], use_gpu)
            index.add(embs)
            kept.extend(start + p for p in positions)
        # Keep row order aligned with the index even if some images were unreadable.
        self.image_products = self.image_products.iloc[kept].reset_index(drop=True)
        return index

    def _hits(self, products: pd.DataFrame, scores, positions) -> list[SearchHit]:
        hits = []
        for rank, (score, pos) in enumerate(zip(scores, positions), start=1):
            if pos == -1:  # FAISS pads with -1 when fewer than k vectors exist
                continue
            row = products.iloc[pos]
            image_path = self.image_dir / f"{row['random_key']}.jpg"
            hits.append(SearchHit(
                rank=rank,
                score=float(score),
                persian_name=str(row["persian_name"]),
                category_name=str(row["category_name"]),
                brand_name=NO_BRAND if row["brand_id"] == -1 else str(row["brand_name"]),
                image_path=str(image_path) if image_path.is_file() else None,
            ))
        return hits

    def search_text(self, query: str) -> list[SearchHit]:
        if not query or not query.strip():
            return []
        scores, positions = self.text_index.search(self.text_encoder.encode([query], mode="query"), self.top_k)
        return self._hits(self.text_products, scores[0], positions[0])

    def search_image(self, image: Image.Image | None) -> list[SearchHit]:
        if image is None or self.image_index is None:
            return []
        scores, positions = self.image_index.search(self.image_encoder.encode([image.convert("RGB")]), self.top_k)
        return self._hits(self.image_products, scores[0], positions[0])
