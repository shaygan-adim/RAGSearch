"""Raw tables -> cleaned products / queries."""

from __future__ import annotations

import pandas as pd
import pytest

from ragsearch.config import Config, ProductTextConfig, QueryTextConfig
from ragsearch.data.products import build_products
from ragsearch.data.queries import build_queries
from ragsearch.data.text_cleaning import TextCleaner


def test_cleaner_handles_missing_values_and_length():
    cleaner = TextCleaner(use_hazm=False)
    assert cleaner.normalize(None) == ""
    assert cleaner.normalize(float("nan")) == ""
    assert cleaner.normalize("  hi  ") == "hi"
    assert cleaner.long_enough("ab", 2) and not cleaner.long_enough("a", 2) and not cleaner.long_enough(None, 1)


def test_hazm_unifies_arabic_characters():
    pytest.importorskip("hazm")
    assert TextCleaner(use_hazm=True).normalize("علي") == "علی"  # Arabic yeh -> Persian yeh


def test_products_text_with_brand(raw_dir):
    out = build_products(raw_dir, ProductTextConfig(include_brand=True), TextCleaner(use_hazm=False))
    texts = dict(zip(out["random_key"], out["cleaned_text"]))

    assert texts["p1"] == "carpet red Acme"  # name + brand
    assert texts["p2"] == "lamp"  # brand_id == -1 -> no brand appended
    assert texts["p3"] == "Acme"  # empty name, brand only
    assert "p4" not in texts  # empty name and no brand -> dropped
    assert list(out.columns) == ["random_key", "persian_name", "category_id", "category_name",
                                 "brand_id", "brand_name", "cleaned_text", "image_url"]
    assert out.set_index("random_key").loc["p1", "category_name"] == "Home"


def test_products_text_with_category_and_no_brand(raw_dir):
    cfg = ProductTextConfig(include_brand=False, include_category=True, separator=" | ")
    out = build_products(raw_dir, cfg, TextCleaner(use_hazm=False))
    assert out.set_index("random_key").loc["p1", "cleaned_text"] == "carpet red | Home"


def test_queries_aggregation_and_filters(raw_dir):
    out = build_queries(raw_dir, QueryTextConfig(), 2, TextCleaner(use_hazm=False))

    assert out["uid"].tolist() == ["u1"]  # u2 short query, u3 inconsistent view, u4 no view
    row = out.iloc[0]
    assert row["viewed_base_products"] == ["p1"]
    assert row["result_base_products"] == ["p1", "p2", "p5"]  # union over both result pages, de-duplicated
    assert row["cleaned_query"] == "red carpet"
    assert row["category_name"] == "Home"


def test_queries_append_category(raw_dir):
    out = build_queries(raw_dir, QueryTextConfig(append_category=True, separator=" / "), 2, TextCleaner(use_hazm=False))
    assert out.iloc[0]["cleaned_query"] == "red carpet / Home"


def test_missing_raw_table_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        build_products(tmp_path, ProductTextConfig(), TextCleaner(use_hazm=False))


def test_full_stage_writes_parquet(raw_dir, tmp_path):
    from ragsearch.data import preprocess

    cfg = Config()
    cfg.paths.raw_data_dir = str(raw_dir)
    cfg.paths.processed_data_dir = str(tmp_path / "processed")
    cfg.preprocessing.hazm_normalize = False
    preprocess.run(cfg)

    assert len(pd.read_parquet(cfg.paths.products_path)) == 4
    assert len(pd.read_parquet(cfg.paths.queries_path)) == 1
