"""Typed configuration, loaded from a single YAML file (``configs/default.yaml``).

Every pipeline stage reads its own section. Defaults here mirror the YAML file
(a test enforces that), so a missing key falls back to the documented value.
Relative paths are resolved against the current working directory, so run the
CLI from the repository root.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


@dataclass
class PathsConfig:
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    image_dir: str = "data/images"  # one `<random_key>.jpg` per product
    mlm_model_dir: str = "artifacts/mlm_model"  # output of `ragsearch mlm`
    checkpoint_path: str = "artifacts/retrieval_system_full.pth"  # output of `ragsearch contrastive`
    results_dir: str = "results"

    @property
    def products_path(self) -> Path:
        return Path(self.processed_data_dir) / "preprocessed_products.parquet"

    @property
    def queries_path(self) -> Path:
        return Path(self.processed_data_dir) / "preprocessed_queries.parquet"

    @property
    def text_results_dir(self) -> Path:
        return Path(self.results_dir) / "text_only"

    @property
    def image_results_dir(self) -> Path:
        return Path(self.results_dir) / "image_only"


@dataclass
class QueryTextConfig:
    append_category: bool = False  # "<query><sep><category name>"
    separator: str = " "


@dataclass
class ProductTextConfig:
    # Text is built as: name [+ category] [+ brand]
    include_category: bool = False
    include_brand: bool = True
    separator: str = " "


@dataclass
class PreprocessingConfig:
    query_text: QueryTextConfig = field(default_factory=QueryTextConfig)
    product_text: ProductTextConfig = field(default_factory=ProductTextConfig)
    min_query_length: int = 2  # drop queries shorter than this after cleaning
    hazm_normalize: bool = True


@dataclass
class ModelConfig:
    """Architecture of the text dual encoder (shared by training, evaluation and the app)."""

    max_length: int = 64
    projection_dim: int = 128
    dropout: float = 0.1


@dataclass
class MLMConfig:
    base_model_id: str = "HooshvareLab/distilbert-fa-zwnj-base"
    output_dir: str = "artifacts/mlm_checkpoints"  # Trainer checkpoints
    test_size: float = 0.1
    max_length: int = 128
    mlm_probability: float = 0.15
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    save_total_limit: int = 2
    logging_steps: int = 100
    seed: int = 42


@dataclass
class ContrastiveConfig:
    base_model: str | None = None  # defaults to `paths.mlm_model_dir`
    test_size: float = 0.1
    val_size: float = 0.1
    seed: int = 42
    # Negative sampling mix; the remainder (1 - type1 - type2) is "random product".
    neg_shown_not_clicked_ratio: float = 0.4
    neg_same_category_ratio: float = 0.3
    batch_size: int = 32
    val_batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    temperature: float = 0.07
    lambda_cls: float = 0.1  # weight of the auxiliary category-classification loss
    log_steps: int = 50


@dataclass
class TextExperiment:
    """One text-retrieval system to evaluate.

    ``type`` is one of:
      * ``sentence_transformer`` - any sentence-transformers model id/path,
      * ``hf_automodel``         - a HF encoder pooled with ``pooling`` (cls | mean),
      * ``custom_dual``          - the trained dual encoder; ``path`` is the .pth checkpoint
                                   and ``base_path`` the HF dir used for tokenizer/architecture.
    ``max_length: null`` keeps the model's own default (sentence-transformers only).
    """

    name: str
    type: str
    path: str
    pooling: str = "cls"
    max_length: int | None = 64
    base_path: str | None = None


def _default_experiments() -> list[TextExperiment]:
    return [
        TextExperiment("Heydari-Persian-Emb", "sentence_transformer", "heydariAI/persian-embeddings",
                       pooling="default", max_length=None),
        TextExperiment("Hooshvare-DistilBert", "hf_automodel", "HooshvareLab/distilbert-fa-zwnj-base"),
        TextExperiment("Custom-MLM-Base", "hf_automodel", "artifacts/mlm_model"),
        TextExperiment("Custom-Dual-Contrastive", "custom_dual", "artifacts/retrieval_system_full.pth",
                       base_path="artifacts/mlm_model"),
    ]


@dataclass
class TextEvalConfig:
    k_values: list[int] = field(default_factory=lambda: [3, 5, 10, 20, 40])
    batch_size: int = 64
    index_batch_size: int = 10_000  # products encoded per chunk while building the index
    query_subset_ratio: float = 0.5  # fraction of queries kept (stratified by category)
    test_size: float = 0.5  # validation / test split of the kept queries
    seed: int = 42
    use_faiss_gpu: bool = True  # silently falls back to CPU when unavailable
    experiments: list[TextExperiment] = field(default_factory=_default_experiments)


@dataclass
class ImageEvalConfig:
    models: dict[str, str] = field(default_factory=lambda: {
        "clip": "openai/clip-vit-base-patch16",
        "convnextv2": "facebook/convnextv2-base-22k-224",
        "siglip2": "google/siglip2-so400m-patch16-naflex",
        "dinov3": "facebook/dinov3-vits16-pretrain-lvd1689m",
    })
    k_values: list[int] = field(default_factory=lambda: [3, 5, 10, 20, 40])
    batch_size: int = 64
    data_sample_fraction: float = 1.0  # fraction of query sessions kept
    test_size: float = 0.5  # validation / test split of unique query texts
    augmentation_k: int = 5  # shown-but-not-clicked products added per query in the "augmented" set
    seed: int = 42
    use_faiss_gpu: bool = True


@dataclass
class AppConfig:
    text_base_model_id: str = "HooshvareLab/distilbert-fa-zwnj-base"
    image_model_id: str = "google/siglip2-so400m-patch16-naflex"
    batch_size: int = 32
    top_k: int = 40
    page_size: int = 10
    use_faiss_gpu: bool = True


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mlm: MLMConfig = field(default_factory=MLMConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    text_eval: TextEvalConfig = field(default_factory=TextEvalConfig)
    image_eval: ImageEvalConfig = field(default_factory=ImageEvalConfig)
    app: AppConfig = field(default_factory=AppConfig)


def _build(cls: type[T], data: Any, where: str) -> T:
    """Recursively instantiate dataclass ``cls`` from a plain dict, rejecting unknown keys."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config section '{where}' must be a mapping, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} in config section '{where}' (valid: {sorted(known)})")

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        if is_dataclass(hint):
            kwargs[name] = _build(hint, value, f"{where}.{name}")
        elif get_origin(hint) is list and is_dataclass(get_args(hint)[0]):
            item_cls = get_args(hint)[0]
            kwargs[name] = [_build(item_cls, v, f"{where}.{name}[{i}]") for i, v in enumerate(value or [])]
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config at ``path`` (``None`` returns the built-in defaults)."""
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _build(Config, raw, "config")
