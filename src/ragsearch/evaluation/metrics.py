"""Ranking metrics with binary relevance, averaged over queries."""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Sequence


def compute_retrieval_metrics(
    retrieved: Sequence[Sequence[Hashable]],
    relevant: Sequence[Collection[Hashable]],
    k_values: Sequence[int],
) -> dict[str, float]:
    """Mean MRR, NDCG and Precision/Recall/F1 @ each k.

    Args:
        retrieved: per query, the ranked list of retrieved ids (best first, at least ``max(k_values)`` long
            for the cut-offs to be meaningful).
        relevant: per query, the ids that count as correct.
        k_values: cut-offs for P@k, R@k and F1@k.

    Queries with no relevant ids are skipped. MRR and NDCG are computed over the top
    ``max(k_values)`` results. Precision always divides by ``k`` (even if fewer results exist).

    Returns:
        ``{"MRR", "NDCG", "P@k", "R@k", "F1@k", ...}`` (all zeros if there is no scoreable query).
    """
    max_k = max(k_values)
    totals = {"MRR": 0.0, "NDCG": 0.0}
    for k in k_values:
        totals.update({f"P@{k}": 0.0, f"R@{k}": 0.0, f"F1@{k}": 0.0})

    n_queries = 0
    for ranked, truth in zip(retrieved, relevant):
        truth = set(truth)
        if not truth:
            continue
        n_queries += 1
        ranked = list(ranked)[:max_k]
        hit_flags = [r in truth for r in ranked]

        first_hit = next((i for i, hit in enumerate(hit_flags) if hit), None)
        if first_hit is not None:
            totals["MRR"] += 1.0 / (first_hit + 1)

        dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(hit_flags) if hit)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(truth), max_k)))
        totals["NDCG"] += dcg / idcg

        for k in k_values:
            hits = len(set(ranked[:k]) & truth)
            precision, recall = hits / k, hits / len(truth)
            f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
            totals[f"P@{k}"] += precision
            totals[f"R@{k}"] += recall
            totals[f"F1@{k}"] += f1

    if n_queries == 0:
        return totals
    return {name: value / n_queries for name, value in totals.items()}
