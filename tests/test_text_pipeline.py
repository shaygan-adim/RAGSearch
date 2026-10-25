"""Dual encoder, contrastive data/loss, MLM corpus mixing and the text evaluation plumbing."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")

from conftest import OneHotTextEncoder  # noqa: E402
from ragsearch.config import Config, ContrastiveConfig, ModelConfig, TextExperiment  # noqa: E402
from ragsearch.data.catalog import ProductCatalog  # noqa: E402
from ragsearch.data.splits import split_sessions  # noqa: E402
from ragsearch.models.dual_encoder import CLASSIFIER_WEIGHT_KEY, RetrievalSystem, load_retrieval_system  # noqa: E402
from ragsearch.training import contrastive  # noqa: E402


def make_products(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "random_key": [f"k{i}" for i in range(n)],
        "cleaned_text": [f"item{i} red" for i in range(n)],
        "category_id": [10 + i % 3 for i in range(n)],  # 3 categories
    })


def make_sessions(catalog: ProductCatalog, n: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        viewed = [f"k{i}", f"k{(i + 3) % 30}"]
        shown = viewed + [f"k{(i + 6) % 30}", f"k{(i + 9) % 30}", "unknown-key"]
        rows.append({"cleaned_query": f"red item{i}", "category_id": 10 + i % 3, "viewed": viewed, "shown": shown})
    df = pd.DataFrame(rows)
    df["cat_idx"] = df["category_id"].map(catalog.encode_category)
    return df


# --- catalogue / negative sampling ---------------------------------------------------------

def test_catalog_encodes_categories():
    catalog = ProductCatalog.from_frame(make_products())
    assert catalog.num_classes == 3 and len(catalog) == 30
    assert catalog.encode_category(0) == -1 and catalog.encode_category(999) == -1
    assert sorted(catalog.category_to_idx.values()) == [0, 1, 2]
    assert all(catalog.category_ids[p] == 10 for p in catalog.by_category[10])


def test_negative_sampling_never_returns_a_viewed_product():
    catalog = ProductCatalog.from_frame(make_products())
    cfg = ContrastiveConfig()
    rng = random.Random(0)
    viewed = {0, 3}
    shown_not_viewed = [6, 9]
    for _ in range(500):
        neg = contrastive.sample_negative(rng, 0, viewed, shown_not_viewed, catalog, cfg)
        assert neg not in viewed


def test_negative_strategies_honour_their_ratios():
    catalog = ProductCatalog.from_frame(make_products())
    rng = random.Random(1)

    only_shown = ContrastiveConfig(neg_shown_not_clicked_ratio=1.0, neg_same_category_ratio=0.0)
    assert {contrastive.sample_negative(rng, 0, {0}, [6, 9], catalog, only_shown) for _ in range(100)} <= {6, 9}

    only_category = ContrastiveConfig(neg_shown_not_clicked_ratio=0.0, neg_same_category_ratio=1.0)
    same = {contrastive.sample_negative(rng, 0, {0}, [6, 9], catalog, only_category) for _ in range(100)}
    assert same <= set(catalog.by_category[10]) - {0}

    # No shown-but-not-viewed candidates: falls back to a random product instead of failing.
    assert contrastive.sample_negative(rng, 0, {0}, [], catalog, only_shown) != 0


# --- dataset, model, loss -------------------------------------------------------------------

@pytest.fixture
def tiny_setup(tiny_model_dir):
    from transformers import AutoTokenizer

    catalog = ProductCatalog.from_frame(make_products())
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    return catalog, tokenizer, tiny_model_dir


def test_triplet_dataset_items(tiny_setup):
    catalog, tokenizer, _ = tiny_setup
    ds = contrastive.TripletDataset(make_sessions(catalog), catalog, tokenizer, ContrastiveConfig(), max_length=8, seed=0)

    assert len(ds) == 12 * 2  # one triplet per (session, viewed product)
    item = ds[0]
    assert item["q_input_ids"].shape == (8,) and item["pos_attention_mask"].shape == (8,)
    assert item["pos_cat_idx"].item() in range(3) and item["neg_cat_idx"].item() in range(3)
    assert ds.samples[0][2] != ds.samples[0][3]  # positive != negative


def test_dataset_is_reproducible_for_a_seed(tiny_setup):
    catalog, tokenizer, _ = tiny_setup
    sessions = make_sessions(catalog)
    a = contrastive.TripletDataset(sessions, catalog, tokenizer, ContrastiveConfig(), 8, seed=5)
    b = contrastive.TripletDataset(sessions, catalog, tokenizer, ContrastiveConfig(), 8, seed=5)
    assert a.samples == b.samples


def test_loss_is_finite_and_backpropagates(tiny_setup):
    catalog, tokenizer, model_dir = tiny_setup
    cfg = ContrastiveConfig()
    ds = contrastive.TripletDataset(make_sessions(catalog), catalog, tokenizer, cfg, max_length=8, seed=0)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=8)))

    model = RetrievalSystem(model_dir, catalog.num_classes, embedding_dim=16)
    loss, contrastive_loss, cls_loss = contrastive.compute_loss(model, batch, cfg)
    assert torch.isfinite(loss) and contrastive_loss.item() > 0 and cls_loss.item() > 0
    assert loss.item() == pytest.approx(contrastive_loss.item() + cfg.lambda_cls * cls_loss.item(), rel=1e-5)

    loss.backward()
    assert model.query_encoder.contrastive_head[0].weight.grad is not None
    assert model.product_encoder.bert.embeddings.word_embeddings.weight.grad is not None


def test_queries_without_category_are_masked_in_classification_loss(tiny_setup):
    catalog, tokenizer, model_dir = tiny_setup
    cfg = ContrastiveConfig()
    ds = contrastive.TripletDataset(make_sessions(catalog), catalog, tokenizer, cfg, max_length=8, seed=0)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=8)))
    batch["q_cat_idx"] = torch.full_like(batch["q_cat_idx"], -1)  # no query has a category

    model = RetrievalSystem(model_dir, catalog.num_classes, embedding_dim=16).eval()
    loss, _, cls_loss = contrastive.compute_loss(model, batch, cfg)  # must not raise or produce NaN
    assert torch.isfinite(loss) and torch.isfinite(cls_loss)


def test_embeddings_are_unit_norm_and_checkpoint_roundtrips(tiny_setup, tmp_path):
    catalog, tokenizer, model_dir = tiny_setup
    model = RetrievalSystem(model_dir, num_classes=3, embedding_dim=16).eval()

    # Checkpoint layout must stay compatible with the original notebooks' state_dicts.
    keys = set(model.state_dict())
    assert CLASSIFIER_WEIGHT_KEY in keys
    assert "query_encoder.contrastive_head.0.weight" in keys
    assert "product_encoder.bert.embeddings.word_embeddings.weight" in keys

    ckpt = tmp_path / "ckpt.pth"
    torch.save(model.state_dict(), ckpt)
    restored = load_retrieval_system(ckpt, model_dir, embedding_dim=16)  # num_classes inferred from the file

    inputs = tokenizer(["red tv", "blue lamp"], padding=True, return_tensors="pt")
    with torch.no_grad():
        a = model.encode_query(inputs["input_ids"], inputs["attention_mask"])
        b = restored.encode_query(inputs["input_ids"], inputs["attention_mask"])
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a.norm(dim=1), torch.ones(2), atol=1e-5)


def test_loading_a_foreign_checkpoint_fails_clearly(tiny_model_dir, tmp_path):
    ckpt = tmp_path / "other.pth"
    torch.save({"w": torch.zeros(1)}, ckpt)
    with pytest.raises(KeyError, match="not a RetrievalSystem checkpoint"):
        load_retrieval_system(ckpt, tiny_model_dir)


# --- text encoders --------------------------------------------------------------------------

def test_text_encoders_return_normalised_float32(tiny_model_dir):
    from ragsearch.models.text_encoders import build_text_encoder

    for pooling in ("cls", "mean"):
        spec = TextExperiment("t", "hf_automodel", str(tiny_model_dir), pooling=pooling, max_length=8)
        encoder = build_text_encoder(spec, ModelConfig(), batch_size=2, device="cpu")
        embs = encoder.encode(["red tv", "blue lamp", "sofa"], mode="query")
        assert embs.shape == (3, 32) and embs.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-5)

    with pytest.raises(ValueError, match="pooling"):
        build_text_encoder(TextExperiment("t", "hf_automodel", str(tiny_model_dir), pooling="max"),
                           ModelConfig(), 2, "cpu")
    with pytest.raises(ValueError, match="unknown type"):
        build_text_encoder(TextExperiment("t", "nope", "x"), ModelConfig(), 2, "cpu")


def test_custom_dual_spec_uses_separate_towers(tiny_model_dir, tmp_path):
    from ragsearch.models.text_encoders import build_text_encoder

    torch.manual_seed(0)
    model = RetrievalSystem(tiny_model_dir, num_classes=3, embedding_dim=16)
    ckpt = tmp_path / "dual.pth"
    torch.save(model.state_dict(), ckpt)

    spec = TextExperiment("dual", "custom_dual", str(ckpt), max_length=8, base_path=str(tiny_model_dir))
    encoder = build_text_encoder(spec, ModelConfig(projection_dim=16), batch_size=4, device="cpu")
    q = encoder.encode(["red tv"], mode="query")
    p = encoder.encode(["red tv"], mode="product")
    assert q.shape == p.shape == (1, 16)
    assert not np.allclose(q, p)  # different towers -> different embeddings for the same text

    with pytest.raises(ValueError, match="base_path"):
        build_text_encoder(TextExperiment("dual", "custom_dual", str(ckpt)), ModelConfig(), 4, "cpu")


# --- MLM corpus -----------------------------------------------------------------------------

def test_mlm_mix_is_balanced_and_deterministic():
    from ragsearch.training.mlm import mix_texts

    products = [f"p{i}" for i in range(50)]
    queries = [f"q{i}" for i in range(10)]
    mixed = mix_texts(products, queries, seed=1)
    assert len(mixed) == 20
    assert sum(t.startswith("q") for t in mixed) == 10 and sum(t.startswith("p") for t in mixed) == 10
    assert mixed == mix_texts(products, queries, seed=1)
    assert mixed != mix_texts(products, queries, seed=2)


def test_mlm_dataset_feeds_the_masking_collator(tiny_model_dir):
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling

    from ragsearch.training.mlm import TextDataset

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    ds = TextDataset(["red tv", "blue lamp", "big sofa"], tokenizer, max_length=8)
    assert len(ds) == 3 and len(ds[0]["input_ids"]) == 8
    assert tokenizer.unk_token_id not in ds[0]["input_ids"]  # the fixture vocabulary really covers these words

    batch = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.5)([ds[i % 3] for i in range(30)])
    assert batch["input_ids"].shape == batch["labels"].shape == (30, 8)
    assert (batch["labels"] != -100).any()  # something was masked
    assert "special_tokens_mask" not in batch


# --- text evaluation plumbing ---------------------------------------------------------------

def test_text_eval_finds_the_right_products():
    pytest.importorskip("faiss")
    from ragsearch.evaluation import text_eval

    n = 40
    products = pd.DataFrame({
        "random_key": [f"k{i}" for i in range(n)],
        "cleaned_text": [f"item{i}" for i in range(n)],
        "category_id": [1 + i % 2 for i in range(n)],
    })
    # Each session's query text names the product it viewed, so a one-hot encoder is a perfect retriever.
    queries = pd.DataFrame({
        "cleaned_query": [f"item{i}" for i in range(n)],
        "category_id": [1 + i % 2 for i in range(n)],
        "viewed_base_products": [np.array([f"k{i}"]) for i in range(n)],
        "result_base_products": [np.array([f"k{i}", f"k{(i + 1) % n}"]) for i in range(n)],
    })

    cfg = Config()
    cfg.text_eval.query_subset_ratio = 1.0
    cfg.text_eval.k_values = [1, 3]
    cfg.text_eval.index_batch_size = 16  # several chunks, exercises incremental indexing
    cfg.text_eval.use_faiss_gpu = False

    data = text_eval.prepare_data(products, queries, cfg)
    ccfg = cfg.contrastive
    train, val, held_out = split_sessions(queries, ccfg.test_size, ccfg.val_size, ccfg.seed)
    assert len(data.validation) + len(data.test) == len(held_out)
    assert data.keys == [f"k{i}" for i in range(n)]  # the pool still covers every session
    evaluated = set(data.validation["cleaned_query"]) | set(data.test["cleaned_query"])
    assert evaluated == set(held_out["cleaned_query"])
    assert not evaluated & (set(train["cleaned_query"]) | set(val["cleaned_query"]))
    assert not set(data.validation["cleaned_query"]) & set(data.test["cleaned_query"])

    metrics = text_eval.evaluate_split(OneHotTextEncoder(n), data, data.test, cfg)
    assert metrics["MRR"] == pytest.approx(1.0) and metrics["NDCG"] == pytest.approx(1.0)
    assert metrics["R@1"] == pytest.approx(1.0) and metrics["P@3"] == pytest.approx(1 / 3)
