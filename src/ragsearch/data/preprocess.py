"""``ragsearch preprocess``: raw parquet tables -> cleaned products and queries."""

from __future__ import annotations

import logging

from ragsearch.config import Config
from ragsearch.data.products import build_products
from ragsearch.data.queries import build_queries
from ragsearch.data.text_cleaning import TextCleaner

logger = logging.getLogger(__name__)


def run(cfg: Config) -> None:
    raw_dir = cfg.paths.raw_data_dir
    cleaner = TextCleaner(use_hazm=cfg.preprocessing.hazm_normalize)
    cfg.paths.products_path.parent.mkdir(parents=True, exist_ok=True)

    products = build_products(raw_dir, cfg.preprocessing.product_text, cleaner)
    products.to_parquet(cfg.paths.products_path, index=False)
    logger.info("Saved %s", cfg.paths.products_path)

    queries = build_queries(raw_dir, cfg.preprocessing.query_text, cfg.preprocessing.min_query_length, cleaner)
    queries.to_parquet(cfg.paths.queries_path, index=False)
    logger.info("Saved %s", cfg.paths.queries_path)
