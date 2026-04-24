"""Generate a one-line catalog summary for a newly-ingested document.

Used by the ingestion pipeline when the uploader didn't provide one on
the form. The output is stored on the document record as ``summary`` and
read by the Studio classifier's catalog builder.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from app.services.llm import LLMService


_MAX_MARKDOWN_CHARS = 4000
_MAX_SUMMARY_TOKENS = 120


def _load_prompt() -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / "ingestion_summary.txt"
    return path.read_text(encoding="utf-8")


class SummaryGenerator:
    """One-shot LLM wrapper that turns (title, markdown) into a one-line brief."""

    def __init__(self, llm: LLMService | None = None, prompt: str | None = None):
        self._llm = llm or LLMService()
        self._prompt = prompt or _load_prompt()

    async def generate(
        self,
        *,
        title: str,
        markdown: str,
        language: str = "en",
    ) -> str:
        """Return a one-line summary, or ``""`` on any failure.

        Never raises — ingestion must continue even if the LLM call dies.
        """
        if not title.strip():
            return ""

        snippet = (markdown or "").strip()[:_MAX_MARKDOWN_CHARS]
        user_payload = (
            f"Title: {title}\n"
            f"Language: {language}\n"
            f"Text (truncated):\n{snippet}"
        )

        try:
            raw = await asyncio.to_thread(
                self._llm.call,
                prompt=user_payload,
                system_instruction=self._prompt,
                max_tokens=_MAX_SUMMARY_TOKENS,
                add_warning=False,
            )
            return (raw or "").strip().splitlines()[0].strip() if raw else ""
        except Exception as exc:
            logger.warning("SummaryGenerator failed for title={!r}: {}", title, exc)
            return ""
