"""``ragsearch eval-image``: compare image encoders on product image -> similar products.

There is no image query in the logs, so the task is built from co-clicks: a product image is the
query and the other products that users opened for the same text query are the relevant items.
Two variants are scored:

* **pure**      - only clicked products (groups of >= 2 clicks);
* **augmented** - clicks plus up to ``augmentation_k`` random shown-but-not-clicked products.

Query texts (not products) are split into validation / test so no text query appears in both.
Models are ranked on ``Val_Augmented`` (falling back to ``Val_Pure``); the winner is run on test.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from ragsearch.config import Config, ImageEvalConfig
from ragsearch.data.splits import impute_categories, stratified_subsample
from ragsearch.evaluation.index import FlatIPIndex
from ragsearch.evaluation.metrics import compute_retrieval_metrics
from ragsearch.models.image_encoder import ImageEncoder, load_images
from ragsearch.utils import as_id_list, resolve_device, set_seed

logger = logging.getLogger(__name__)

PAIR_COLUMNS = ["query_key", "ground_truth", "original_text_query"]


@dataclass
class ImageEvalData:
    candidate_keys: list[str]
    val_pure: pd.DataFrame
    val_augmented: pd.DataFrame
    test_pure: pd.DataFrame
    test_augmented: pd.DataFrame


def build_pairs(
    clicks_by_query: Mapping[str, set[str]],
    shown_by_query: Mapping[str, set[str]],
    augmentation_k: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn per-text click/shown sets into (pure, augmented) rows of ``query_key -> ground_truth``.

    Every product in a group takes a turn as the query; the others are its ground truth.
    """
    pure, augmented = [], []

    def add_group(rows: list[dict], members: Sequence[str], text: str) -> None:
        for i, key in enumerate(members):
            rows.append({
                "query_key": key,
                "ground_truth": [m for j, m in enumerate(members) if j != i],
                "original_text_query": text,
            })

    for text, clicks in clicks_by_query.items():
        if len(clicks) >= 2:
            add_group(pure, sorted(clicks), text)

        # Seeded per text and drawn from a sorted list, so the sample is reproducible across runs.
        extras = sorted(shown_by_query.get(text, set()) - clicks)
        rng = random.Random(seed + len(text))
        group = sorted(clicks) + rng.sample(extras, min(len(extras), augmentation_k))
        if len(group) >= 2:
            add_group(augmented, group, text)

    return pd.DataFrame(pure, columns=PAIR_COLUMNS), pd.DataFrame(augmented, columns=PAIR_COLUMNS)


def available_image_keys(image_dir: str | Path) -> set[str]:
    return {p.stem.strip() for p in Path(image_dir).glob("*.jpg")}


def prepare_data(products: pd.DataFrame, queries: pd.DataFrame, image_keys: set[str], cfg: ImageEvalConfig) -> ImageEvalData:
    """Restrict everything to products that have an image, then build and split the pair sets."""
    products = products.assign(random_key=products["random_key"].astype(str).str.strip())
    products = products[products["random_key"].isin(image_keys)]
    valid = set(products["random_key"])
    logger.info("Products with an image: %d", len(valid))

    queries = queries.copy()
    queries["valid_clicks"] = queries["viewed_base_products"].map(lambda v: sorted({k for k in as_id_list(v) if k in valid}))
    queries["valid_shown"] = queries["result_base_products"].map(lambda v: sorted({k for k in as_id_list(v) if k in valid}))
    queries = queries[queries["valid_clicks"].map(len) > 0].reset_index(drop=True)
    queries["text"] = queries["cleaned_query"].astype(str).str.lower().str.strip()
    logger.info("Query sessions with a usable click: %d", len(queries))

    if cfg.data_sample_fraction < 1.0:
        queries = impute_categories(queries, "valid_clicks", dict(zip(products["random_key"], products["category_id"])))
        queries = stratified_subsample(queries, cfg.data_sample_fraction, "stratify_cat", cfg.seed)

    clicks: dict[str, set[str]] = {}
    shown: dict[str, set[str]] = {}
    for text, c, s in zip(queries["text"], queries["valid_clicks"], queries["valid_shown"]):
        clicks.setdefault(text, set()).update(c)
        shown.setdefault(text, set()).update(s)

    pure, augmented = build_pairs(clicks, shown, cfg.augmentation_k, cfg.seed)
    logger.info("Pure: %d pairs, augmented: %d pairs (from %d unique query texts)", len(pure), len(augmented), len(clicks))

    val_texts, test_texts = train_test_split(sorted(clicks), test_size=cfg.test_size, random_state=cfg.seed)

    def split(pairs: pd.DataFrame, texts: list[str]) -> pd.DataFrame:
        return pairs[pairs["original_text_query"].isin(set(texts))].copy()

    return ImageEvalData(
        candidate_keys=products["random_key"].tolist(),
        val_pure=split(pure, val_texts),
        val_augmented=split(augmented, val_texts),
        test_pure=split(pure, test_texts),
        test_augmented=split(augmented, test_texts),
    )


def build_index(encoder, keys: Sequence[str], image_dir: str | Path, batch_size: int, use_gpu: bool):
    """Embed the candidate images. Returns the index and the keys actually indexed (unreadable files are skipped)."""
    index: FlatIPIndex | None = None
    indexed: list[str] = []
    for start in tqdm(range(0, len(keys), batch_size), desc="Indexing images"):
        chunk = keys[start : start + batch_size]
        images, positions = load_images(image_dir, chunk)
        if not images:
            continue
        embs = encoder.encode(images)
        if index is None:
            index = FlatIPIndex(embs.shape[1], use_gpu=use_gpu)
        index.add(embs)
        indexed.extend(chunk[p] for p in positions)
    if index is None:
        raise RuntimeError("No candidate image could be read; check paths.image_dir")
    logger.info("Index built with %d vectors", len(index))
    return index, indexed


def evaluate_pairs(encoder, index: FlatIPIndex, indexed_keys: list[str], pairs: pd.DataFrame, image_dir, cfg: ImageEvalConfig) -> dict[str, float]:
    """Search with each pair's query image (excluding itself) and score against its ground truth."""
    k_max = max(cfg.k_values)
    query_keys = pairs["query_key"].tolist()
    retrieved: list[list[str]] = [[] for _ in query_keys]  # stays empty when the query image is unreadable

    for start in tqdm(range(0, len(query_keys), cfg.batch_size), desc="Querying"):
        chunk = query_keys[start : start + cfg.batch_size]
        images, positions = load_images(image_dir, chunk)
        if not images:
            continue
        _, top = index.search(encoder.encode(images), k_max + 1)  # +1: the query itself is in the index
        for pos, row in zip(positions, top):
            hits = [indexed_keys[i] for i in row if i != -1 and indexed_keys[i] != chunk[pos]]
            retrieved[start + pos] = hits[:k_max]

    return compute_retrieval_metrics(retrieved, pairs["ground_truth"].tolist(), cfg.k_values)


def _evaluate_model(model_key: str, model_id: str, data: ImageEvalData, stages: dict[str, pd.DataFrame],
                    cfg: Config, device: str) -> list[dict]:
    icfg = cfg.image_eval
    encoder = ImageEncoder(model_key, model_id, device).load()
    try:
        index, indexed = build_index(encoder, data.candidate_keys, cfg.paths.image_dir, icfg.batch_size, icfg.use_faiss_gpu)
        rows = []
        for stage, pairs in stages.items():
            if pairs.empty:
                logger.warning("Skipping %s: no pairs", stage)
                continue
            logger.info("Evaluating %s on %s (%d queries)", model_key, stage, len(pairs))
            metrics = evaluate_pairs(encoder, index, indexed, pairs, cfg.paths.image_dir, icfg)
            logger.info("%s -> NDCG %.4f | MRR %.4f", stage, metrics["NDCG"], metrics["MRR"])
            rows.append({**metrics, "model": model_key, "stage": stage})
        index.close()
        return rows
    finally:
        encoder.close()


def run(cfg: Config) -> None:
    icfg = cfg.image_eval
    set_seed(icfg.seed)
    device = resolve_device()

    image_keys = available_image_keys(cfg.paths.image_dir)
    if not image_keys:
        raise FileNotFoundError(f"No .jpg files in {cfg.paths.image_dir}; see data/README.md")
    products = pd.read_parquet(cfg.paths.products_path)
    queries = pd.read_parquet(cfg.paths.queries_path)
    data = prepare_data(products, queries, image_keys, icfg)
    del products, queries

    out_path = cfg.paths.image_results_dir / "metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_stages = {"Val_Pure": data.val_pure, "Val_Augmented": data.val_augmented}
    test_stages = {"Test_Pure": data.test_pure, "Test_Augmented": data.test_augmented}

    results: list[dict] = []
    for model_key, model_id in icfg.models.items():
        logger.info("=== %s (%s) ===", model_key, model_id)
        try:
            results += _evaluate_model(model_key, model_id, data, val_stages, cfg, device)
        except Exception:  # noqa: BLE001 - e.g. gated model without HF_TOKEN; keep going with the others
            logger.exception("Skipping %s", model_key)
            continue
        pd.DataFrame(results).to_csv(out_path, index=False)  # checkpoint after every model

    table = pd.DataFrame(results)
    if table.empty:
        raise RuntimeError("No model finished; nothing to select.")
    stage = "Val_Augmented" if (table["stage"] == "Val_Augmented").any() else "Val_Pure"
    best = table[table["stage"] == stage].sort_values("NDCG", ascending=False).iloc[0]
    logger.info("Best model on %s: %s (NDCG %.4f)", stage, best["model"], best["NDCG"])

    results += _evaluate_model(best["model"], icfg.models[best["model"]], data, test_stages, cfg, device)
    pd.DataFrame(results).to_csv(out_path, index=False)
    logger.info("Wrote %s", out_path)
