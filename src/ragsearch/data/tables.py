"""Helpers for reading the raw parquet tables and joining lookup tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(raw_dir: str | Path, name: str) -> pd.DataFrame:
    """Read ``<raw_dir>/<name>.parquet`` with an actionable error if it is missing."""
    path = Path(raw_dir) / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw table {path}. See data/README.md for the expected layout.")
    return pd.read_parquet(path)


def attach_title(df: pd.DataFrame, lookup: pd.DataFrame, key_col: str, out_col: str) -> pd.DataFrame:
    """Left-join ``lookup.title`` (matched on ``lookup.id``) onto ``df[key_col]`` as ``out_col``.

    The lookup ids must be unique, otherwise the join would silently duplicate rows.
    """
    titles = lookup[["id", "title"]].rename(columns={"id": key_col, "title": out_col})
    return df.merge(titles, on=key_col, how="left", validate="many_to_one")
