"""Config, utils, metrics, index, splits - no ML dependencies needed."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT
from ragsearch.config import Config, load_config
from ragsearch.data.splits import consolidate_rare, impute_categories, stratified_split, stratified_subsample
from ragsearch.utils import as_id_list, flatten_unique, most_common_category


# --- config ----------------------------------------------------------------------------------

def test_default_yaml_matches_code_defaults():
    assert load_config(REPO_ROOT / "configs" / "default.yaml") == Config()


def test_unknown_config_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("mlm:\n  learning_rte: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="learning_rte"):
        load_config(bad)


def test_partial_yaml_falls_back_to_defaults(tmp_path):
    f = tmp_path / "partial.yaml"
    f.write_text("model:\n  max_length: 32\n", encoding="utf-8")
    cfg = load_config(f)
    assert cfg.model.max_length == 32
    assert cfg.model.projection_dim == Config().model.projection_dim


# --- utils -----------------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (np.array(["a", "b"]), ["a", "b"]),
    (["a", " b "], ["a", "b"]),
    ({"a"}, ["a"]),
    ("['a', 'b']", ["a", "b"]),
    ("['a' 'b']", ["a", "b"]),  # numpy-style repr: no comma
    ("[a b]", ["a", "b"]),
    (None, []),
    (float("nan"), []),
])
def test_as_id_list(value, expected):
    assert as_id_list(value) == expected


def test_flatten_unique_is_sorted_union():
    assert flatten_unique([["b", "a"], ["c", "a"]]) == ["a", "b", "c"]


def test_most_common_category_ignores_unknown_and_zero():
    cats = {"p1": 5, "p2": 5, "p3": 7, "p4": 0}
    assert most_common_category(["p1", "p2", "p3", "p4", "zz"], cats) == 5
    assert most_common_category(["p4", "zz"], cats) == 0


# --- metrics ---------------------------------------------------------------------------------

def test_metrics_match_hand_computation():
    from ragsearch.evaluation.metrics import compute_retrieval_metrics

    retrieved = [["x", "a", "y", "b"], ["x", "y", "z", "w"], ["a"]]
    relevant = [{"a", "b"}, {"a"}, set()]  # third query has no ground truth -> skipped
    m = compute_retrieval_metrics(retrieved, relevant, k_values=[2, 4])

    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert m["MRR"] == pytest.approx((1 / 2 + 0) / 2)
    assert m["NDCG"] == pytest.approx((dcg / idcg + 0) / 2)
    assert m["P@2"] == pytest.approx((1 / 2 + 0) / 2)
    assert m["R@2"] == pytest.approx((1 / 2 + 0) / 2)
    assert m["F1@2"] == pytest.approx((0.5 + 0) / 2)
    assert m["P@4"] == pytest.approx((2 / 4 + 0) / 2)
    assert m["R@4"] == pytest.approx((1.0 + 0) / 2)
    assert m["F1@4"] == pytest.approx((2 * 0.5 * 1.0 / 1.5 + 0) / 2)


def test_metrics_perfect_and_empty():
    from ragsearch.evaluation.metrics import compute_retrieval_metrics

    perfect = compute_retrieval_metrics([["a", "b", "c"]], [{"a"}], [1, 3])
    assert perfect["MRR"] == perfect["NDCG"] == perfect["P@1"] == perfect["R@3"] == 1.0

    empty = compute_retrieval_metrics([["a"]], [set()], [1])
    assert empty["MRR"] == 0.0 and empty["NDCG"] == 0.0


# --- index (needs faiss) -----------------------------------------------------------------------

def test_flat_ip_index_returns_nearest_first():
    pytest.importorskip("faiss")
    from ragsearch.evaluation.index import FlatIPIndex

    index = FlatIPIndex(3, use_gpu=True)  # GPU request must silently fall back on a CPU-only build
    index.add(np.eye(3, dtype=np.float32))
    scores, positions = index.search(np.array([[0, 1, 0]], dtype=np.float32), k=5)
    assert positions[0][0] == 1 and scores[0][0] == pytest.approx(1.0)
    assert positions[0][3] == -1  # fewer vectors than k -> padded with -1
    assert len(index) == 3


# --- result logger ----------------------------------------------------------------------------

def test_result_logger_header_and_row(tmp_path):
    from ragsearch.evaluation.results import ResultLogger

    log = ResultLogger(tmp_path / "out" / "metrics.csv", [3, 5])
    log.log("m", "cls", 64, "Validation", {"P@3": 0.123456, "MRR": 0.5})
    df = pd.read_csv(tmp_path / "out" / "metrics.csv")
    assert list(df.columns) == ["Timestamp", "Model_Name", "Pooling", "Max_Len", "Split",
                                "P@3", "R@3", "F1@3", "P@5", "R@5", "F1@5", "MRR", "NDCG"]
    assert df.loc[0, "P@3"] == 0.1235 and df.loc[0, "MRR"] == 0.5 and df.loc[0, "R@3"] == 0


def test_committed_results_use_the_logger_schema():
    df = pd.read_csv(REPO_ROOT / "results" / "text_only" / "metrics.csv")
    assert {"Model_Name", "Split", "NDCG", "MRR"} <= set(df.columns)


# --- splits ------------------------------------------------------------------------------------

def test_stratified_split_preserves_classes_and_falls_back():
    df = pd.DataFrame({"c": [0] * 20 + [1] * 20, "x": range(40)})
    train, test = stratified_split(df, 0.5, "c", seed=0)
    assert test["c"].value_counts().to_dict() == {0: 10, 1: 10}

    rare = pd.DataFrame({"c": [0] * 10 + [1], "x": range(11)})  # class 1 has a single member
    a, b = stratified_split(rare, 0.5, "c", seed=0)
    assert len(a) + len(b) == 11


def test_stratified_subsample_keeps_fraction():
    df = pd.DataFrame({"c": [0] * 20 + [1] * 20})
    assert len(stratified_subsample(df, 0.5, "c", seed=0)) == 20
    assert stratified_subsample(df, 1.0, "c", seed=0) is df


def test_consolidate_rare_and_impute():
    assert consolidate_rare(pd.Series([1, 1, 2, 3, 3])).tolist() == [1, 1, -1, 3, 3]

    df = pd.DataFrame({"category_id": [0, 4], "clicked": [["p1", "p2"], ["p1"]]})
    out = impute_categories(df, "clicked", {"p1": 9, "p2": 9})
    assert out["stratify_cat"].tolist() == [9, 4]
