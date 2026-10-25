"""Small helpers shared across the pipeline stages."""

from __future__ import annotations

import logging
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


_QUOTED = re.compile(r"['\"]([^'\"]+)['\"]")


def configure_logging(level: int = logging.INFO) -> None:
    """Show ``level`` for ragsearch's own logs; third-party libraries stay at WARNING."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("ragsearch").setLevel(level)


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str = "auto") -> str:
    """``"auto"`` -> CUDA when available, else CPU."""
    if preferred != "auto":
        return preferred
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def as_id_list(value: Any) -> list[str]:
    """Normalise a list-like cell of product keys to ``list[str]``.

    Parquet round-trips Python lists as numpy arrays; plain lists/sets are accepted too. Stringified
    lists (``"['a', 'b']"`` or numpy-style ``"['a' 'b']"``) are parsed leniently. Anything else
    (None, NaN, unparsable) yields an empty list.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(v).strip() for v in value.tolist()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v).strip() for v in value]
    if isinstance(value, str):
        quoted = _QUOTED.findall(value)
        if quoted:
            return [q.strip() for q in quoted]
        return [t for t in re.split(r"[\s,\[\]]+", value) if t]
    return []


def flatten_unique(lists: Iterable[Iterable[str]]) -> list[str]:
    """Sorted union of several id lists (sorted so results are deterministic)."""
    return sorted({item for sub in lists for item in sub})


def most_common_category(keys: Sequence[str], key_to_category: Mapping[str, int]) -> int:
    """Most frequent non-zero category among ``keys`` (0 if none are known)."""
    cats = [key_to_category.get(k, 0) for k in keys]
    cats = [c for c in cats if c != 0]
    return Counter(cats).most_common(1)[0][0] if cats else 0
