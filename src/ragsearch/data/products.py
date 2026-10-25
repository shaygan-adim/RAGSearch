"""Build the cleaned product table."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ragsearch.config import ProductTextConfig
from ragsearch.data.tables import attach_title, read_table
from ragsearch.data.text_cleaning import TextCleaner

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "random_key", "persian_name", "category_id", "category_name",
    "brand_id", "brand_name", "cleaned_text", "image_url",
]

NO_BRAND_ID = -1


def _append(base: pd.Series, extra: pd.Series, mask: pd.Series, separator: str) -> pd.Series:
    """``base + separator + extra`` where ``mask`` is true, ``base`` elsewhere."""
    return base.where(~mask, base + separator + extra.astype(str))


def build_products(raw_dir: str | Path, cfg: ProductTextConfig, cleaner: TextCleaner) -> pd.DataFrame:
    """Join products with category and brand names and build the text used for retrieval.

    Output text is ``persian_name [+ category_name] [+ brand_name]`` (see ``cfg``),
    normalised by ``cleaner``. Products whose final text is empty are dropped.
    """
    tqdm.pandas(desc="Normalising product text")
    logger.info("Building products from %s", raw_dir)

    products = read_table(raw_dir, "base_products")
    df = attach_title(products, read_table(raw_dir, "categories"), "category_id", "category_name")
    df = attach_title(df, read_table(raw_dir, "brands"), "brand_id", "brand_name")

    text = df["persian_name"].fillna("")
    if cfg.include_category:
        text = _append(text, df["category_name"], df["category_name"].notna(), cfg.separator)
    if cfg.include_brand:
        has_brand = df["brand_name"].notna() & (df["brand_id"] != NO_BRAND_ID)
        text = _append(text, df["brand_name"], has_brand, cfg.separator)

    df["cleaned_text"] = text.progress_apply(cleaner.normalize)

    out = df.reindex(columns=OUTPUT_COLUMNS)
    out = out[out["cleaned_text"].str.len() > 0].reset_index(drop=True)
    logger.info("Products: %d raw -> %d kept", len(products), len(out))
    return out
