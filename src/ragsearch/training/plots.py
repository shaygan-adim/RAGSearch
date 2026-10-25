"""Training-curve figures (matplotlib is optional: plotting is skipped with a warning if missing)."""

from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping the training-curve plot")
        return None
    return plt


def plot_mlm_history(log_history: Sequence[dict], title: str, out_path: str | Path) -> None:
    """Training loss per logged step and validation loss per evaluation (from ``Trainer.state.log_history``)."""
    plt = _pyplot()
    if plt is None:
        return
    train = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    evals = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    fig, ax = plt.subplots(figsize=(10, 6))
    if train:
        ax.plot(*zip(*train), label="Training loss", alpha=0.6)
    if evals:
        ax.plot(*zip(*evals), label="Validation loss", color="red", marker="o", linestyle="--")
    ax.set(title=title, xlabel="Step", ylabel="Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, plt, out_path)


def plot_contrastive_history(step_losses: Sequence[float], val_losses: Sequence[float], log_steps: int,
                             out_path: str | Path) -> None:
    plt = _pyplot()
    if plt is None:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))
    left.plot(step_losses, label="Training loss")
    left.set(title="Training loss", xlabel=f"Logged step (every {log_steps})", ylabel="Loss")
    right.plot(range(1, len(val_losses) + 1), val_losses, marker="o", color="orange", label="Validation loss")
    right.set(title="Validation loss per epoch", xlabel="Epoch", ylabel="Loss")
    for ax in (left, right):
        ax.grid(True, alpha=0.3)
        ax.legend()
    _save(fig, plt, out_path)


def _save(fig, plt, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)
