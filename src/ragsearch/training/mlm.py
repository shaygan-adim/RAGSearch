"""``ragsearch mlm``: domain-adapt the Persian DistilBERT with masked-language modelling.

The corpus is an even mix of query texts and product texts: both are undersampled to the size
of the smaller set, concatenated and shuffled. The best checkpoint (lowest validation loss)
is saved to ``paths.mlm_model_dir`` and is the starting point for contrastive training.
"""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from ragsearch.config import Config
from ragsearch.training.plots import plot_mlm_history
from ragsearch.utils import set_seed

logger = logging.getLogger(__name__)


def mix_texts(product_texts: list[str], query_texts: list[str], seed: int) -> list[str]:
    """Balanced 50/50 corpus: undersample both lists to ``min(len)`` and shuffle."""
    rng = np.random.default_rng(seed)
    n = min(len(product_texts), len(query_texts))
    products = rng.choice(np.asarray(product_texts, dtype=object), size=n, replace=False)
    queries = rng.choice(np.asarray(query_texts, dtype=object), size=n, replace=False)
    mixed = np.concatenate([products, queries])
    rng.shuffle(mixed)
    return mixed.tolist()


def _warmup_kwargs(ratio: float) -> dict[str, float]:
    """Warmup as a fraction of training: ``warmup_ratio`` (transformers 4.x) or float ``warmup_steps`` (5.x)."""
    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}


class TextDataset(Dataset):
    """Tokenises lazily, so memory stays flat however large the corpus is."""

    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.texts, self.tokenizer, self.max_length = texts, tokenizer, max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict[str, list[int]]:
        enc = self.tokenizer(self.texts[i], padding="max_length", truncation=True, max_length=self.max_length,
                             return_special_tokens_mask=True)
        return {key: enc[key] for key in ("input_ids", "attention_mask", "special_tokens_mask")}


def run(cfg: Config) -> None:
    mcfg = cfg.mlm
    set_seed(mcfg.seed)

    products = pd.read_parquet(cfg.paths.products_path)
    queries = pd.read_parquet(cfg.paths.queries_path)
    texts = mix_texts(
        products["cleaned_text"].dropna().astype(str).tolist(),
        queries["cleaned_query"].dropna().astype(str).tolist(),
        mcfg.seed,
    )
    logger.info("MLM corpus: %d texts (%d products + %d queries available)", len(texts), len(products), len(queries))

    tokenizer = AutoTokenizer.from_pretrained(mcfg.base_model_id)

    train_texts, val_texts = train_test_split(texts, test_size=mcfg.test_size, random_state=mcfg.seed)
    train_ds = TextDataset(train_texts, tokenizer, mcfg.max_length)
    val_ds = TextDataset(val_texts, tokenizer, mcfg.max_length)
    logger.info("Train: %d, validation: %d", len(train_ds), len(val_ds))

    model = AutoModelForMaskedLM.from_pretrained(mcfg.base_model_id)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mcfg.mlm_probability)
    args = TrainingArguments(
        output_dir=mcfg.output_dir,
        num_train_epochs=mcfg.num_train_epochs,
        per_device_train_batch_size=mcfg.batch_size,
        per_device_eval_batch_size=mcfg.batch_size,
        learning_rate=mcfg.learning_rate,
        weight_decay=mcfg.weight_decay,
        **_warmup_kwargs(mcfg.warmup_ratio),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=mcfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=mcfg.logging_steps,
        seed=mcfg.seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator)

    result = trainer.train()
    logger.info("Training finished: %s", result.metrics)

    trainer.save_model(cfg.paths.mlm_model_dir)
    tokenizer.save_pretrained(cfg.paths.mlm_model_dir)
    logger.info("Saved domain-adapted model to %s", cfg.paths.mlm_model_dir)

    plot_mlm_history(trainer.state.log_history, f"MLM domain adaptation: {mcfg.base_model_id}",
                     cfg.paths.text_results_dir / "mlm_training_curve.png")
