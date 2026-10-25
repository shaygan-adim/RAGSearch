"""Demo search engine (stub encoders), HTML rendering and CLI wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import OneHotTextEncoder
from ragsearch import cli


def test_cli_parses_every_stage():
    parser = cli.build_parser()
    for name in [*cli.STAGES, "app"]:
        args = parser.parse_args([name, "--config", "x.yaml"])
        assert args.command == name and args.config == "x.yaml"
    assert parser.parse_args(["app", "--share", "--port", "9"]).share is True


def test_cli_rejects_unknown_command():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["train-everything"])


def test_every_stage_module_exposes_run():
    import importlib

    for module_name, _ in cli.STAGES.values():
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("faiss")
        assert callable(importlib.import_module(module_name).run)


class ColourImageEncoder:
    def encode(self, images) -> np.ndarray:
        vecs = np.array([np.asarray(i, dtype=np.float32).reshape(-1, 3).mean(axis=0) for i in images])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def engine(tmp_path):
    pytest.importorskip("faiss")
    PIL = pytest.importorskip("PIL.Image")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from ragsearch.app.engine import SearchEngine

    image_dir = tmp_path / "img"
    image_dir.mkdir()
    PIL.new("RGB", (4, 4), (250, 5, 5)).save(image_dir / "k0.jpg")
    PIL.new("RGB", (4, 4), (5, 5, 250)).save(image_dir / "k2.jpg")  # k1 has no image

    products = pd.DataFrame({
        "random_key": ["k0", "k1", "k2"],
        "persian_name": ["<b>name0</b>", "name1", "name2"],
        "category_name": ["c", "c", "c"],
        "brand_id": [-1, 5, 5],
        "brand_name": [None, "Acme", "Acme"],
        "cleaned_text": ["item0", "item1", "item2"],
    })
    return SearchEngine(products, image_dir, OneHotTextEncoder(3), ColourImageEncoder(), top_k=5, batch_size=2,
                        use_gpu=False)


def test_text_search_ranks_by_similarity(engine):
    hits = engine.search_text("item1")
    assert [h.persian_name for h in hits][0] == "name1"
    assert hits[0].rank == 1 and hits[0].score == pytest.approx(1.0)
    assert hits[0].brand_name == "Acme" and hits[0].image_path is None  # k1 has no image
    assert engine.search_text("   ") == []


def test_no_brand_label(engine):
    assert engine.search_text("item0")[0].brand_name == "No Brand"


def test_image_search_only_covers_products_with_images(engine):
    PIL = pytest.importorskip("PIL.Image")
    hits = engine.search_image(PIL.new("RGB", (4, 4), (255, 0, 0)))
    assert [h.persian_name for h in hits] == ["<b>name0</b>", "name2"]  # k1 absent; red query -> k0 first
    assert hits[0].image_path.endswith("k0.jpg")
    assert engine.search_image(None) == []


def test_text_cards_escape_html(engine):
    pytest.importorskip("gradio")
    from ragsearch.app.gradio_app import render_text_cards

    html = render_text_cards(engine.search_text("item0"))
    assert "&lt;b&gt;name0&lt;/b&gt;" in html and "<b>name0</b>" not in html
    assert "No results" in render_text_cards([])


def test_pagination_helpers():
    pytest.importorskip("gradio")
    from ragsearch.app.gradio_app import page_slice, total_pages

    assert total_pages(0, 10) == 1 and total_pages(40, 10) == 4 and total_pages(41, 10) == 5
    assert page_slice(list(range(25)), 3, 10) == [20, 21, 22, 23, 24]


def test_demo_builds(engine):
    pytest.importorskip("gradio")
    from ragsearch.app.gradio_app import build_demo

    demo = build_demo(engine, page_size=2)
    assert demo is not None
