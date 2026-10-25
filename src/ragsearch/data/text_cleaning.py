"""Persian text normalisation."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans free text; optionally applies Hazm normalisation.

    Hazm unifies Arabic/Persian character variants, fixes spacing around
    punctuation and handles zero-width non-joiners.
    """

    def __init__(self, use_hazm: bool = True):
        self.use_hazm = use_hazm
        self._normalizer = None
        if use_hazm:
            try:
                import hazm
            except ImportError as exc:
                raise ImportError(
                    "Hazm normalisation needs the 'preprocess' extra: pip install -e \".[preprocess]\" "
                    "(or set preprocessing.hazm_normalize: false)"
                ) from exc
            self._normalizer = hazm.Normalizer()

    def normalize(self, text: object) -> str:
        """Return the cleaned string (``""`` for None/NaN/blank input)."""
        if text is None or pd.isna(text):
            return ""
        text = str(text).strip()
        if not text:
            return ""
        if self._normalizer is not None:
            try:
                text = self._normalizer.normalize(text)
            except Exception:  # noqa: BLE001 - never let one odd string kill a 1M-row job
                logger.debug("Hazm failed on %r; keeping the stripped text", text, exc_info=True)
        return text

    @staticmethod
    def long_enough(text: object, min_len: int) -> bool:
        """True if ``text`` is a string with at least ``min_len`` characters."""
        return isinstance(text, str) and len(text) >= min_len
