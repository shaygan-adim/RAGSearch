"""Build the cleaned query table: one row per search session (``uid``)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ragsearch.config import QueryTextConfig
from ragsearch.data.tables import attach_title, read_table
from ragsearch.data.text_cleaning import TextCleaner
from ragsearch.utils import as_id_list, flatten_unique

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "uid", "query", "cleaned_query", "result_base_products",
    "category_id", "category_name", "viewed_base_products",
]


def _log_drop(what: str, before: int, after: int) -> None:
    pct = 100 * (before - after) / before if before else 0.0
    logger.info("Dropped %d rows (%.2f%%): %s", before - after, pct, what)


def build_queries(
    raw_dir: str | Path,
    cfg: QueryTextConfig,
    min_query_length: int,
    cleaner: TextCleaner,
) -> pd.DataFrame:
    """Aggregate raw search pages and product views into one row per ``uid``.

    Columns:
      * ``result_base_products`` - every product shown for the query (all result pages),
      * ``viewed_base_products`` - products the user opened (the positives),
    both de-duplicated. Sessions are dropped when nothing was viewed, when a viewed
    product was never shown (inconsistent log), or when the cleaned query is too short.
    """
    tqdm.pandas(desc="Normalising queries")
    logger.info("Building queries from %s", raw_dir)

    searches = read_table(raw_dir, "searches")
    views = read_table(raw_dir, "base_views")
    categories = read_table(raw_dir, "categories")

    # One row per search session; a session spans several result pages.
    sessions = (
        searches.groupby("uid")
        .agg(
            query=("query", "first"),
            category_id=("category_id", "first"),
            result_base_products=("result_base_product_rks", lambda s: flatten_unique(as_id_list(x) for x in s)),
        )
        .reset_index()
    )

    # Views reference a search *page* id; map them to the session uid.
    page_to_uid = searches[["id", "uid"]].drop_duplicates().rename(columns={"id": "search_id"})
    viewed = (
        views[["search_id", "base_product_rk"]]
        .merge(page_to_uid, on="search_id", how="inner")
        .groupby("uid")["base_product_rk"]
        .agg(lambda s: sorted(set(s)))
        .rename("viewed_base_products")
        .reset_index()
    )

    df = sessions.merge(viewed, on="uid", how="left")
    df["viewed_base_products"] = df["viewed_base_products"].map(lambda v: v if isinstance(v, list) else [])

    before = len(df)
    df = df[df["viewed_base_products"].map(len) > 0]
    _log_drop("no viewed products", before, len(df))

    # A user cannot open a product that was not returned: such sessions are log noise.
    before = len(df)
    consistent = [set(v) <= set(r) for v, r in zip(df["viewed_base_products"], df["result_base_products"])]
    df = df[consistent]
    _log_drop("viewed products not among the results", before, len(df))

    df = attach_title(df, categories, "category_id", "category_name")

    text = df["query"].fillna("")
    if cfg.append_category:
        has_category = (df["category_id"] != 0) & df["category_id"].notna() & df["category_name"].notna()
        text = text.where(~has_category, text + cfg.separator + df["category_name"].astype(str))
    df["cleaned_query"] = text.progress_apply(cleaner.normalize)

    before = len(df)
    df = df[df["cleaned_query"].map(lambda q: cleaner.long_enough(q, min_query_length))]
    _log_drop(f"cleaned query shorter than {min_query_length} characters", before, len(df))

    out = df[OUTPUT_COLUMNS].reset_index(drop=True)
    logger.info("Queries: %d sessions kept", len(out))
    return out
