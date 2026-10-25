"""Command-line entry point: ``ragsearch <command> [--config configs/default.yaml]``.

Commands run in pipeline order:

    preprocess   raw parquet tables         -> cleaned products + queries
    mlm          cleaned texts              -> domain-adapted DistilBERT
    contrastive  adapted model + sessions   -> query/product dual-encoder checkpoint
    eval-text    text models                -> results/text_only/metrics.csv
    eval-image   image models               -> results/image_only/metrics.csv
    app          checkpoint + product data  -> Gradio demo

Heavy dependencies (torch, transformers, faiss, gradio) are imported lazily, so
``ragsearch preprocess`` only needs the core install.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from collections.abc import Sequence

from ragsearch.config import Config, load_config
from ragsearch.utils import configure_logging

DEFAULT_CONFIG = "configs/default.yaml"

# command -> (module with a `run(cfg)` function, help text)
STAGES = {
    "preprocess": ("ragsearch.data.preprocess", "Clean the raw tables into products/queries parquet files."),
    "mlm": ("ragsearch.training.mlm", "Domain-adapt DistilBERT with masked-language modelling."),
    "contrastive": ("ragsearch.training.contrastive", "Train the query/product dual encoder."),
    "eval-text": ("ragsearch.evaluation.text_eval", "Compare text retrievers and report the best on test."),
    "eval-image": ("ragsearch.evaluation.image_eval", "Compare image encoders and report the best on test."),
}


def _run_app(cfg: Config, share: bool, port: int | None) -> None:
    from ragsearch.app.engine import SearchEngine
    from ragsearch.app.gradio_app import build_demo

    engine = SearchEngine.from_config(cfg)
    demo = build_demo(engine, cfg.app.page_size)
    demo.launch(share=share, server_port=port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragsearch", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in STAGES.items():
        sub.add_parser(name, help=help_text)
    app = sub.add_parser("app", help="Launch the Gradio demo.")
    app.add_argument("--share", action="store_true", help="Create a public gradio.live link.")
    app.add_argument("--port", type=int, default=None, help="Server port (default: Gradio's choice).")
    for p in sub.choices.values():
        p.add_argument("--config", default=DEFAULT_CONFIG, help=f"YAML config (default: {DEFAULT_CONFIG}).")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(logging.INFO)
    cfg = load_config(args.config)

    if args.command == "app":
        _run_app(cfg, args.share, args.port)
    else:
        importlib.import_module(STAGES[args.command][0]).run(cfg)


if __name__ == "__main__":
    main()
