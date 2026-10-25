"""In-memory product catalogue built from ``preprocessed_products.parquet``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ProductCatalog:
    """Column-oriented view of the products, indexed by position.

    ``key_to_idx`` maps a product ``random_key`` to its position; ``category_idx`` holds the
    label-encoded category (0..num_classes-1) used by the auxiliary classification head.
    """

    keys: list[str]
    texts: list[str]
    category_ids: np.ndarray
    category_idx: np.ndarray
    key_to_idx: dict[str, int]
    by_category: dict[int, list[int]]  # raw category id -> positions
    category_to_idx: dict[int, int]  # raw category id -> label-encoded id

    @property
    def num_classes(self) -> int:
        return len(self.category_to_idx)

    def __len__(self) -> int:
        return len(self.keys)

    @classmethod
    def from_frame(cls, products: pd.DataFrame) -> ProductCatalog:
        keys = products["random_key"].astype(str).str.strip().tolist()
        texts = products["cleaned_text"].fillna("").astype(str).tolist()
        category_ids = products["category_id"].to_numpy()

        by_category: dict[int, list[int]] = {}
        for pos, cat in enumerate(category_ids.tolist()):
            by_category.setdefault(cat, []).append(pos)
        category_to_idx = {cat: i for i, cat in enumerate(sorted(by_category))}

        return cls(
            keys=keys,
            texts=texts,
            category_ids=category_ids,
            category_idx=np.array([category_to_idx[c] for c in category_ids.tolist()], dtype=np.int64),
            key_to_idx={k: i for i, k in enumerate(keys)},
            by_category=by_category,
            category_to_idx=category_to_idx,
        )

    def encode_category(self, category_id: int) -> int:
        """Label-encoded id of ``category_id``; ``-1`` for 0 ("no category") or unseen categories."""
        if category_id == 0:
            return -1
        return self.category_to_idx.get(category_id, -1)
