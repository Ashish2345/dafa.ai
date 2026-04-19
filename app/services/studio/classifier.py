"""Stage-1 document selector for Studio queries.

Picks specific documents from a catalog, not categories.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from loguru import logger

from app.services.llm import LLMService


SourceType = Literal["acts-rules", "finance-acts", "nrb", "ird", "gazette", "najirs"]
ALL_TYPES: tuple[SourceType, ...] = (
    "acts-rules",
    "finance-acts",
    "nrb",
    "ird",
    "gazette",
    "najirs",
)


@dataclass
class CatalogEntry:
    id: str
    name: str
    category: SourceType
    brief: str = ""


@dataclass
class ClassifierResult:
    selected_doc_ids: list[str]
    reasoning: str
    used_fallback: bool                     # True if LLM failed / empty catalog and we broadened
    catalog_size: int = 0


def _load_prompt() -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "studio_classifier.txt"
    return path.read_text(encoding="utf-8")


def _catalog_to_json(catalog: Iterable[CatalogEntry]) -> list[dict]:
    return [
        {"id": e.id, "name": e.name, "category": e.category, "brief": e.brief or ""}
        for e in catalog
    ]


class TypeClassifier:
    """Wraps an LLMService call that selects specific documents for a query."""

    def __init__(self, llm: LLMService, prompt: str | None = None):
        self._llm = llm
        self._prompt = prompt or _load_prompt()

    async def classify(
        self,
        query: str,
        catalog: list[CatalogEntry],
        history: Iterable[tuple[str, str]] | None = None,
        pinned_types: list[SourceType] | None = None,
    ) -> ClassifierResult:
        """Return selected document IDs for the query.

        pinned_types narrows the catalog before calling the LLM.
        """
        # Honor pin by filtering catalog to pinned types only.
        if pinned_types:
            scoped = [e for e in catalog if e.category in pinned_types]
        else:
            scoped = list(catalog)

        if not scoped:
            return ClassifierResult(
                selected_doc_ids=[],
                reasoning="empty-catalog",
                used_fallback=True,
                catalog_size=0,
            )

        # Short-circuit: if the scope has only one document, just pick it.
        if len(scoped) == 1:
            return ClassifierResult(
                selected_doc_ids=[scoped[0].id],
                reasoning="only-one-in-scope",
                used_fallback=False,
                catalog_size=1,
            )

        user_payload = {
            "query": query,
            "history": [{"role": r, "content": c} for r, c in (history or [])][-3:],
            "catalog": _catalog_to_json(scoped),
        }

        try:
            raw = await asyncio.to_thread(
                self._llm.call,
                prompt=json.dumps(user_payload, ensure_ascii=False),
                system_instruction=self._prompt,
                max_tokens=512,
                response_mime_type="application/json",
                add_warning=False,
            )
            parsed = json.loads(raw)
            catalog_ids = {e.id for e in scoped}
            selected = [sid for sid in (parsed.get("selected") or []) if sid in catalog_ids]
            reasoning = str(parsed.get("reasoning", "")).strip() or "llm-selection"
            return ClassifierResult(
                selected_doc_ids=selected,
                reasoning=reasoning,
                used_fallback=False,
                catalog_size=len(scoped),
            )
        except Exception as exc:
            logger.warning("Studio classifier failed, falling back to all scoped docs: {}", exc)
            return ClassifierResult(
                selected_doc_ids=[e.id for e in scoped],
                reasoning="classifier-unavailable",
                used_fallback=True,
                catalog_size=len(scoped),
            )
