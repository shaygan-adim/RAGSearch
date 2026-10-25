"""Shared fixtures: synthetic raw tables, and a tiny offline BERT + tokenizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A miniature version of the raw parquet tables.

    Sessions (uid):
      u1  kept: two result pages, one product viewed
      u2  dropped: cleaned query shorter than 2 chars
      u3  dropped: viewed product p3 was never in the results
      u4  dropped: nothing viewed
    """
    d = tmp_path / "raw"
    d.mkdir()

    pd.DataFrame({
        "image_url": ["u1", "u2", "u3", "u4", "u5"],
        "random_key": ["p1", "p2", "p3", "p4", "p5"],
        "category_id": [1, 2, 2, 2, 1],
        "brand_id": [10, -1, 10, -1, 99],
        "english_name": [""] * 5,
        "persian_name": ["carpet red", "lamp", "", "", "sofa"],
    }).to_parquet(d / "base_products.parquet")
    pd.DataFrame({"id": [1, 2], "title": ["Home", "Lighting"]}).to_parquet(d / "categories.parquet")
    pd.DataFrame({"id": [10, 99], "title": ["Acme", "Zed"]}).to_parquet(d / "brands.parquet")

    pd.DataFrame({
        "id": ["s1", "s2", "s3", "s4", "s5"],
        "uid": ["u1", "u1", "u2", "u3", "u4"],
        "query": ["red carpet", "red carpet", "x", "lamp", "sofa"],
        "category_id": [1, 1, 0, 2, 1],
        "result_base_product_rks": [
            np.array(["p1", "p2"]), np.array(["p5", "p2"]), np.array(["p1"]), np.array(["p2"]), np.array(["p5"]),
        ],
    }).to_parquet(d / "searches.parquet")
    pd.DataFrame({
        "id": ["v1", "v2", "v3"],
        "search_id": ["s1", "s3", "s4"],
        "base_product_rk": ["p1", "p1", "p3"],
    }).to_parquet(d / "base_views.parquet")
    return d


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    """A 1-layer DistilBERT with a small word-level vocab, saved to disk (no network needed)."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    d = tmp_path_factory.mktemp("tiny_bert")
    words = ["red", "blue", "green", "tv", "lamp", "sofa", "carpet", "big", "small", "item"]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    # Built with `tokenizers` directly: BertTokenizerFast's vocab argument differs between transformers 4 and 5
    # (v5 silently ignores `vocab_file`, which turned every word into [UNK]).
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    special = {"pad_token": "[PAD]", "unk_token": "[UNK]", "cls_token": "[CLS]", "sep_token": "[SEP]",
               "mask_token": "[MASK]"}
    tk = Tokenizer(models.WordLevel({tok: i for i, tok in enumerate(vocab)}, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                      special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    transformers.PreTrainedTokenizerFast(tokenizer_object=tk, **special).save_pretrained(d)

    config = transformers.DistilBertConfig(vocab_size=len(vocab), dim=32, hidden_dim=64, n_layers=1, n_heads=2,
                                           max_position_embeddings=64)
    transformers.DistilBertModel(config).save_pretrained(d)
    return d


class OneHotTextEncoder:
    """Maps the text ``item<i>`` to the i-th unit vector, so ranking behaviour is fully predictable."""

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts, mode: str = "query") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, int(text.replace("item", ""))] = 1.0
        return out
