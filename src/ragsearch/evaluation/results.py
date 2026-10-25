"""Append-only CSV log of evaluation runs."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence

logger = logging.getLogger(__name__)


class ResultLogger:
    """Writes one row per (model, split) with P/R/F1 @ each k plus MRR and NDCG.

    Header: ``Timestamp, Model_Name, Pooling, Max_Len, Split, P@k, R@k, F1@k ..., MRR, NDCG``
    (the schema of ``results/text_only/metrics.csv``). Metric values are rounded to 4 decimals.
    """

    def __init__(self, path: str | Path, k_values: Sequence[int]):
        self.path = Path(path)
        self.metric_columns = [f"{m}@{k}" for k in k_values for m in ("P", "R", "F1")]
        self.header = ["Timestamp", "Model_Name", "Pooling", "Max_Len", "Split", *self.metric_columns, "MRR", "NDCG"]

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.header)

    def log(self, model_name: str, pooling: str, max_length: int | str, split: str, metrics: dict[str, float]) -> None:
        row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_name, pooling, max_length, split]
        row += [f"{metrics.get(col, 0.0):.4f}" for col in self.metric_columns]
        row += [f"{metrics.get('MRR', 0.0):.4f}", f"{metrics.get('NDCG', 0.0):.4f}"]
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        logger.info("Logged %s / %s to %s", model_name, split, self.path)
