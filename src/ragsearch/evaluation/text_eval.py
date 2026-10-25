"""``ragsearch eval-text``: compare text retrievers on query -> product search.

Protocol:
  1. keep sessions with at least one viewed product, subsample them (stratified by category),
  2. the candidate pool is every product that was viewed or shown for a kept session,
  3. keep the sessions in the contrastive test split, then split them 50/50 into validation and test (stratified),
  4. rank each configured model on validation, pick the best by NDCG,
  5. report the winner once on the test split.
Ground truth for a query is the set of products the user viewed.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import pandas as pd
import torch
from tqdm.auto import tqdm

from ragsearch.config import Config, TextExperiment
from ragsearch.data.splits import (
    consolidate_rare,
    impute_categories,
    split_sessions,
    stratified_split,
    stratified_subsample,
)
from ragsearch.evaluation.index import FlatIPIndex
from ragsearch.evaluation.metrics import compute_retrieval_metrics
from ragsearch.evaluation.results import ResultLogger
from ragsearch.models.text_encoders import TextEncoder, build_text_encoder
from ragsearch.utils import as_id_list, flatten_unique, resolve_device, set_seed

logger = logging.getLogger(__name__)


@dataclass
class TextEvalData:
    keys: list[str]  # candidate pool, in index order
    texts: list[str]
    key_to_idx: dict[str, int]
    validation: pd.DataFrame
    test: pd.DataFrame


def prepare_data(products: pd.DataFrame, queries: pd.DataFrame, cfg: Config) -> TextEvalData:
    """Filter, subsample and split the preprocessed tables (steps 1-3 above)."""
    ecfg = cfg.text_eval
    queries = queries.copy()
    tqdm.pandas(desc="Parsing id lists")
    for col in ("viewed_base_products", "result_base_products"):
        queries[col] = queries[col].progress_apply(as_id_list)

    queries = queries[queries["viewed_base_products"].map(len) > 0]
    logger.info("Sessions with at least one view: %d", len(queries))

    ccfg = cfg.contrastive
    _, _, held_out = split_sessions(queries, ccfg.test_size, ccfg.val_size, ccfg.seed)

    key_to_category = dict(zip(products["random_key"], products["category_id"]))
    queries = impute_categories(queries, "viewed_base_products", key_to_category)
    queries["stratify_cat"] = consolidate_rare(queries["stratify_cat"])

    if ecfg.query_subset_ratio < 1.0:
        queries = stratified_subsample(queries, ecfg.query_subset_ratio, "stratify_cat", ecfg.seed)
        queries["stratify_cat"] = consolidate_rare(queries["stratify_cat"])
        logger.info("Subsampled to %d sessions", len(queries))

    pool_keys = set(flatten_unique(queries["viewed_base_products"])) | set(flatten_unique(queries["result_base_products"]))
    pool = products[products["random_key"].isin(pool_keys)]
    logger.info("Candidate pool: %d -> %d products", len(products), len(pool))

    queries = queries[queries.index.isin(held_out.index)].copy()
    queries["stratify_cat"] = consolidate_rare(queries["stratify_cat"])
    validation, test = stratified_split(queries, ecfg.test_size, "stratify_cat", ecfg.seed)
    logger.info("Validation: %d sessions, test: %d sessions", len(validation), len(test))

    keys = pool["random_key"].tolist()
    return TextEvalData(
        keys=keys,
        texts=pool["cleaned_text"].tolist(),
        key_to_idx={k: i for i, k in enumerate(keys)},
        validation=validation,
        test=test,
    )


def evaluate_split(encoder: TextEncoder, data: TextEvalData, split: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """Index the candidate pool with the product tower, then search the split's queries."""
    ecfg = cfg.text_eval
    index: FlatIPIndex | None = None
    try:
        for start in tqdm(range(0, len(data.texts), ecfg.index_batch_size), desc="Indexing products"):
            embs = encoder.encode(data.texts[start : start + ecfg.index_batch_size], mode="product")
            if index is None:
                index = FlatIPIndex(embs.shape[1], use_gpu=ecfg.use_faiss_gpu)
            index.add(embs)
            del embs
            gc.collect()

        query_embs = encoder.encode(split["cleaned_query"].tolist(), mode="query")
        _, top = index.search(query_embs, max(ecfg.k_values))
    finally:
        if index is not None:
            index.close()

    relevant = [
        {data.key_to_idx[k] for k in viewed if k in data.key_to_idx} for viewed in split["viewed_base_products"]
    ]
    return compute_retrieval_metrics(top.tolist(), relevant, ecfg.k_values)


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pooling_label(spec: TextExperiment) -> str:
    return "default" if spec.type == "sentence_transformer" else spec.pooling


def _max_len_label(spec: TextExperiment) -> int | str:
    return "default" if spec.max_length is None else spec.max_length


def run(cfg: Config) -> None:
    ecfg = cfg.text_eval
    set_seed(ecfg.seed)
    device = resolve_device()

    products = pd.read_parquet(cfg.paths.products_path)
    queries = pd.read_parquet(cfg.paths.queries_path)
    logger.info("Loaded %d products and %d sessions", len(products), len(queries))
    data = prepare_data(products, queries, cfg)
    del products, queries

    results = ResultLogger(cfg.paths.text_results_dir / "metrics.csv", ecfg.k_values)

    best_ndcg, best_spec = -1.0, None
    for spec in ecfg.experiments:
        logger.info("=== Validation: %s ===", spec.name)
        try:
            encoder = build_text_encoder(spec, cfg.model, ecfg.batch_size, device)
            metrics = evaluate_split(encoder, data, data.validation, cfg)
        except Exception:  # noqa: BLE001 - one broken experiment (e.g. missing checkpoint) must not stop the rest
            logger.exception("Skipping experiment %s", spec.name)
            continue
        results.log(spec.name, _pooling_label(spec), _max_len_label(spec), "Validation", metrics)
        if metrics["NDCG"] > best_ndcg:
            best_ndcg, best_spec = metrics["NDCG"], spec
        del encoder
        _free_memory()

    if best_spec is None:
        raise RuntimeError("No experiment finished; nothing to evaluate on the test split.")

    logger.info("Best on validation: %s (NDCG %.4f). Running the test split.", best_spec.name, best_ndcg)
    encoder = build_text_encoder(best_spec, cfg.model, ecfg.batch_size, device)
    metrics = evaluate_split(encoder, data, data.test, cfg)
    results.log(best_spec.name, _pooling_label(best_spec), _max_len_label(best_spec), "TEST_FINAL", metrics)
    for name, value in metrics.items():
        logger.info("%s: %.4f", name, value)
