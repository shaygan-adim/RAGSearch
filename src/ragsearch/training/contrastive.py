"""``ragsearch contrastive``: train the query/product dual encoder.

Every (query, viewed product) pair is a positive. Each positive gets one sampled negative:

  * shown for that query but never viewed   (``neg_shown_not_clicked_ratio``),
  * random product from the positive's category (``neg_same_category_ratio``),
  * random product from the whole catalogue (the remainder, and the fallback for the two above).

Loss = InfoNCE over the batch (the batch's negatives are appended as extra columns)
     + ``lambda_cls`` * category-classification loss on query, positive and negative.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ragsearch.config import Config, ContrastiveConfig
from ragsearch.data.catalog import ProductCatalog
from ragsearch.data.splits import split_sessions
from ragsearch.models.dual_encoder import RetrievalSystem
from ragsearch.training.plots import plot_contrastive_history
from ragsearch.utils import as_id_list, resolve_device, set_seed

logger = logging.getLogger(__name__)

NO_CATEGORY = -1  # query category index when the query has no (known) category


@dataclass
class TrainingData:
    catalog: ProductCatalog
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def prepare_data(products: pd.DataFrame, queries: pd.DataFrame, cfg: ContrastiveConfig) -> TrainingData:
    """Parse id lists, drop sessions without views and split sessions into train / val / test."""
    catalog = ProductCatalog.from_frame(products)

    queries = queries.copy()
    queries["viewed"] = queries["viewed_base_products"].map(as_id_list)
    queries["shown"] = queries["result_base_products"].map(as_id_list)
    queries = queries[queries["viewed"].map(len) > 0].copy()
    queries["cat_idx"] = queries["category_id"].map(catalog.encode_category)

    train, val, test = split_sessions(queries, cfg.test_size, cfg.val_size, cfg.seed)
    logger.info("Sessions: train %d / val %d / test %d; %d product categories", len(train), len(val), len(test),
                catalog.num_classes)
    return TrainingData(catalog, train, val, test)


def _random_not_in(rng: random.Random, pool: list[int] | range, excluded: set[int], tries: int = 20) -> int | None:
    for _ in range(tries):
        candidate = pool[rng.randrange(len(pool))]
        if candidate not in excluded:
            return candidate
    return None


def sample_negative(rng: random.Random, pos: int, viewed: set[int], shown_not_viewed: list[int],
                    catalog: ProductCatalog, cfg: ContrastiveConfig) -> int:
    """Pick one negative product position for the positive ``pos`` (see module docstring)."""
    r = rng.random()
    neg = None
    if r < cfg.neg_shown_not_clicked_ratio:
        if shown_not_viewed:
            neg = rng.choice(shown_not_viewed)
    elif r < cfg.neg_shown_not_clicked_ratio + cfg.neg_same_category_ratio:
        same_category = catalog.by_category[int(catalog.category_ids[pos])]
        neg = _random_not_in(rng, same_category, viewed)

    if neg is None:  # random product, also the fallback when the first two options had no candidate
        everything = range(len(catalog))
        neg = _random_not_in(rng, everything, viewed, tries=1000)
        if neg is None:
            raise RuntimeError("Could not sample a negative: the catalogue is (almost) all positives")
    return neg


class TripletDataset(Dataset):
    """(query, positive product, negative product) triplets, sampled once at construction."""

    def __init__(self, sessions: pd.DataFrame, catalog: ProductCatalog, tokenizer, cfg: ContrastiveConfig,
                 max_length: int, seed: int):
        self.catalog, self.tokenizer, self.max_length = catalog, tokenizer, max_length
        self.samples: list[tuple[str, int, int, int]] = []  # (query text, query cat idx, pos, neg)

        rng = random.Random(seed)
        for query, cat_idx, viewed_keys, shown_keys in zip(
            sessions["cleaned_query"], sessions["cat_idx"], sessions["viewed"], sessions["shown"]
        ):
            viewed = {catalog.key_to_idx[k] for k in viewed_keys if k in catalog.key_to_idx}
            shown_not_viewed = sorted({catalog.key_to_idx[k] for k in shown_keys if k in catalog.key_to_idx} - viewed)
            for pos in sorted(viewed):
                neg = sample_negative(rng, pos, viewed, shown_not_viewed, catalog, cfg)
                self.samples.append((query, int(cat_idx), pos, neg))

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length,
                             return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        query, q_cat, pos, neg = self.samples[i]
        q_ids, q_mask = self._encode(query)
        p_ids, p_mask = self._encode(self.catalog.texts[pos])
        n_ids, n_mask = self._encode(self.catalog.texts[neg])
        return {
            "q_input_ids": q_ids, "q_attention_mask": q_mask, "q_cat_idx": torch.tensor(q_cat, dtype=torch.long),
            "pos_input_ids": p_ids, "pos_attention_mask": p_mask,
            "pos_cat_idx": torch.tensor(int(self.catalog.category_idx[pos]), dtype=torch.long),
            "neg_input_ids": n_ids, "neg_attention_mask": n_mask,
            "neg_cat_idx": torch.tensor(int(self.catalog.category_idx[neg]), dtype=torch.long),
        }


def compute_loss(model: RetrievalSystem, batch: dict[str, torch.Tensor], cfg: ContrastiveConfig):
    """Returns ``(total, contrastive, classification)`` losses for one batch."""
    q_emb, q_logits, pos_emb, pos_logits, neg_emb, neg_logits = model(batch)

    # Row i must score its own positive highest among all positives and all negatives of the batch.
    sim = torch.cat([q_emb @ pos_emb.T, q_emb @ neg_emb.T], dim=1) / cfg.temperature
    contrastive = F.cross_entropy(sim, torch.arange(sim.size(0), device=sim.device))

    cls_pos = F.cross_entropy(pos_logits, batch["pos_cat_idx"])
    cls_neg = F.cross_entropy(neg_logits, batch["neg_cat_idx"])

    # Queries without a category (-1) are masked out of the classification loss.
    q_target = batch["q_cat_idx"]
    q_valid = (q_target != NO_CATEGORY).float()
    cls_q_each = F.cross_entropy(q_logits, q_target.clamp(min=0), reduction="none")
    cls_q = (cls_q_each * q_valid).sum() / (q_valid.sum() + 1e-8)

    classification = cls_pos + cls_neg + cls_q
    return contrastive + cfg.lambda_cls * classification, contrastive, classification


@dataclass
class History:
    step_losses: list[float] = field(default_factory=list)
    train_epoch_losses: list[float] = field(default_factory=list)
    val_epoch_losses: list[float] = field(default_factory=list)


def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(model, loader, optimizer, cfg: ContrastiveConfig, device: str, history: History) -> float:
    model.train()
    total = 0.0
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        optimizer.zero_grad()
        loss, _, _ = compute_loss(model, _to_device(batch, device), cfg)
        loss.backward()
        optimizer.step()
        total += loss.item()
        if step % cfg.log_steps == 0:
            history.step_losses.append(loss.item())
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, cfg: ContrastiveConfig, device: str) -> tuple[float, float, float]:
    model.eval()
    totals = [0.0, 0.0, 0.0]
    for batch in tqdm(loader, desc="Validating"):
        losses = compute_loss(model, _to_device(batch, device), cfg)
        for i, value in enumerate(losses):
            totals[i] += value.item()
    return tuple(t / len(loader) for t in totals)


def run(cfg: Config) -> None:
    ccfg = cfg.contrastive
    set_seed(ccfg.seed)
    device = resolve_device()
    base_model = ccfg.base_model or cfg.paths.mlm_model_dir

    data = prepare_data(pd.read_parquet(cfg.paths.products_path), pd.read_parquet(cfg.paths.queries_path), ccfg)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    max_length = cfg.model.max_length
    train_ds = TripletDataset(data.train, data.catalog, tokenizer, ccfg, max_length, ccfg.seed)
    val_ds = TripletDataset(data.val, data.catalog, tokenizer, ccfg, max_length, ccfg.seed + 1)
    logger.info("Triplets: train %d, val %d", len(train_ds), len(val_ds))
    train_loader = DataLoader(train_ds, batch_size=ccfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=ccfg.val_batch_size, shuffle=False)

    model = RetrievalSystem(base_model, data.catalog.num_classes, cfg.model.projection_dim, cfg.model.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ccfg.learning_rate, weight_decay=ccfg.weight_decay)

    history = History()
    for epoch in range(1, ccfg.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, ccfg, device, history)
        val_loss, val_contrastive, val_cls = validate(model, val_loader, ccfg, device)
        history.train_epoch_losses.append(train_loss)
        history.val_epoch_losses.append(val_loss)
        logger.info("Epoch %d/%d | train %.4f | val %.4f (contrastive %.4f, classification %.4f)",
                    epoch, ccfg.num_epochs, train_loss, val_loss, val_contrastive, val_cls)

    checkpoint = Path(cfg.paths.checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    logger.info("Saved checkpoint to %s", checkpoint)

    out_dir = cfg.paths.text_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "contrastive_history.json").write_text(json.dumps(history.__dict__, indent=2), encoding="utf-8")
    plot_contrastive_history(history.step_losses, history.val_epoch_losses, ccfg.log_steps,
                             out_dir / "contrastive_training_curve.png")
