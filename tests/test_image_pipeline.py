"""Image-evaluation data construction, indexing and scoring (with a stub colour encoder)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sklearn")
PIL = pytest.importorskip("PIL.Image")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from ragsearch.config import ImageEvalConfig  # noqa: E402
from ragsearch.evaluation import image_eval  # noqa: E402


class MeanColourEncoder:
    """Embeds an image as its normalised mean RGB: same-coloured products are near neighbours."""

    def encode(self, images) -> np.ndarray:
        vecs = np.array([np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(axis=0) for img in images])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


COLOURS = {  # two reds, two blues, one green
    "r1": (250, 10, 10), "r2": (240, 30, 10), "b1": (10, 10, 250), "b2": (10, 30, 240), "g1": (10, 250, 10),
}


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for key, rgb in COLOURS.items():
        PIL.new("RGB", (8, 8), rgb).save(d / f"{key}.jpg")
    return d


def test_build_pairs_pure_and_augmented():
    clicks = {"red": {"r1", "r2"}, "solo": {"g1"}, "none-shown": {"b1"}}
    shown = {"red": {"r1", "r2", "b1"}, "solo": {"g1", "b1", "b2"}, "none-shown": {"b1"}}
    pure, aug = image_eval.build_pairs(clicks, shown, augmentation_k=1, seed=0)

    # Pure: only groups with >= 2 clicks; each product takes a turn as the query.
    assert sorted(pure["query_key"]) == ["r1", "r2"]
    assert pure.set_index("query_key").loc["r1", "ground_truth"] == ["r2"]

    # Augmented: "red" gets its click set plus one shown-not-clicked product; "solo" has one click + one extra.
    red = aug[aug["original_text_query"] == "red"]
    assert len(red) == 3 and set(red["query_key"]) == {"r1", "r2", "b1"}
    assert "none-shown" not in set(aug["original_text_query"])  # a single click and nothing extra -> no pair

    # Same inputs -> same sample (the original used set order and was not reproducible).
    _, aug_again = image_eval.build_pairs(clicks, shown, augmentation_k=1, seed=0)
    pd.testing.assert_frame_equal(aug, aug_again)


def test_prepare_data_restricts_to_available_images_and_splits_by_text():
    products = pd.DataFrame({"random_key": list(COLOURS), "category_id": [1, 1, 2, 2, 3]})
    texts = [f"query {i}" for i in range(6)]
    queries = pd.DataFrame({
        "cleaned_query": texts,
        "category_id": [1] * 6,
        "viewed_base_products": [np.array(["r1", "r2", "missing"])] * 3 + [np.array(["b1", "b2"])] * 3,
        "result_base_products": [np.array(["r1", "r2", "g1"])] * 3 + [np.array(["b1", "b2", "g1"])] * 3,
    })
    data = image_eval.prepare_data(products, queries, set(COLOURS), ImageEvalConfig(test_size=0.5, seed=0))

    assert data.candidate_keys == list(COLOURS)
    val_texts = set(data.val_pure["original_text_query"]) | set(data.val_augmented["original_text_query"])
    test_texts = set(data.test_pure["original_text_query"]) | set(data.test_augmented["original_text_query"])
    assert val_texts and test_texts and not val_texts & test_texts
    assert all("missing" not in gt for gt in pd.concat([data.val_pure, data.test_pure])["ground_truth"])


def test_evaluate_pairs_excludes_the_query_and_stays_aligned(image_dir):
    cfg = ImageEvalConfig(k_values=[1, 3], batch_size=2)
    encoder = MeanColourEncoder()
    keys = list(COLOURS) + ["no_file"]  # one candidate has no image file

    index, indexed = image_eval.build_index(encoder, keys, image_dir, batch_size=2, use_gpu=False)
    assert indexed == list(COLOURS) and len(index) == len(COLOURS)  # unreadable candidate skipped, keys stay aligned

    pairs = pd.DataFrame({
        "query_key": ["unreadable", "r1", "b1"],  # the first query has no image: must not shift the others
        "ground_truth": [["r2"], ["r2"], ["b2"]],
        "original_text_query": ["a", "b", "c"],
    })
    metrics = image_eval.evaluate_pairs(encoder, index, indexed, pairs, image_dir, cfg)

    # r1 -> r2 and b1 -> b2 are rank 1; the unreadable query scores 0. R@1 and MRR average over all three.
    assert metrics["MRR"] == pytest.approx(2 / 3)
    assert metrics["R@1"] == pytest.approx(2 / 3)
    assert metrics["P@1"] == pytest.approx(2 / 3)


def test_available_image_keys(image_dir):
    assert image_eval.available_image_keys(image_dir) == set(COLOURS)
