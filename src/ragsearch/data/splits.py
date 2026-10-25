"""Train/validation/test splitting helpers (stratified, with a random fallback)."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd
from sklearn.model_selection import train_test_split

from ragsearch.utils import most_common_category

logger = logging.getLogger(__name__)


def stratified_split(
    df: pd.DataFrame, test_size: float, stratify_col: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df``; stratify on ``stratify_col`` unless a class is too rare, then split randomly."""
    try:
        return train_test_split(df, test_size=test_size, stratify=df[stratify_col], random_state=seed)
    except ValueError:
        logger.warning("Stratified split on '%s' failed (rare classes); using a random split", stratify_col)
        return train_test_split(df, test_size=test_size, random_state=seed)


def split_sessions(
    sessions: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train / validation / test split of sessions, stratified by ``category_id``.

    ``val_size`` is a fraction of what remains after the test split is taken out.
    """
    train, test = stratified_split(sessions, test_size, "category_id", seed)
    train, val = stratified_split(train, val_size, "category_id", seed)
    return train, val, test


def stratified_subsample(df: pd.DataFrame, fraction: float, stratify_col: str, seed: int) -> pd.DataFrame:
    """Keep ``fraction`` of the rows, preserving the ``stratify_col`` distribution when possible."""
    if fraction >= 1.0:
        return df
    kept, _ = stratified_split(df, test_size=1.0 - fraction, stratify_col=stratify_col, seed=seed)
    return kept


def impute_categories(
    df: pd.DataFrame,
    clicked_col: str,
    key_to_category: Mapping[str, int],
    out_col: str = "stratify_cat",
) -> pd.DataFrame:
    """Add ``out_col``: the query's category, or - when it is 0 (unknown) - the dominant
    category of the products the user opened. Used only to stratify splits."""

    def pick(row) -> int:
        if row["category_id"] != 0:
            return row["category_id"]
        return most_common_category(row[clicked_col], key_to_category)

    out = df.copy()
    out[out_col] = out.apply(pick, axis=1)
    return out


def consolidate_rare(series: pd.Series, min_count: int = 2, fill: int = -1) -> pd.Series:
    """Replace values that occur fewer than ``min_count`` times by ``fill`` (stratification needs >= 2)."""
    counts = series.value_counts()
    frequent = counts[counts >= min_count].index
    return series.where(series.isin(frequent), fill)
