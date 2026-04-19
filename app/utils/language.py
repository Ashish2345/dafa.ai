"""Language helpers — tiny utilities shared across query/synthesis flows."""

import re
from typing import Iterable

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def detect_language(chunks: Iterable) -> str:
    """Infer document language from the first few retrieved chunks.

    Any Devanagari content in the leading chunks flips the language to ``ne``;
    everything else falls through to ``en``. Works with either dataclass-like
    objects exposing ``.text`` or plain dicts with a ``"text"`` key.
    """
    for idx, chunk in enumerate(chunks):
        if idx >= 3:
            break
        text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
        if _DEVANAGARI.search(text or ""):
            return "ne"
    return "en"


def detect_language_from_text(text: str) -> str:
    """Infer language from a raw string — ``ne`` if any Devanagari, else ``en``."""
    return "ne" if _DEVANAGARI.search(text or "") else "en"
