"""StudioRouter — orchestrates classifier → aggregator → synthesis."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Iterable

from loguru import logger

from app.services.llm import LLMService
from app.services.studio.aggregator import (
    AggregatedChunk,
    DocumentAggregator,
    DocumentUsage,
)
from app.services.studio.classifier import (
    CatalogEntry,
    SourceType,
    TypeClassifier,
)


def _load_synthesis_prompt() -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "studio_synthesis.txt"
    return path.read_text(encoding="utf-8")


@dataclass
class StudioEvent:
    kind: str
    payload: dict


class StudioRouter:
    def __init__(
        self,
        classifier: TypeClassifier,
        aggregator: DocumentAggregator,
        llm: LLMService,
        synthesis_prompt: str | None = None,
    ):
        self._classifier = classifier
        self._aggregator = aggregator
        self._llm = llm
        self._synthesis_prompt = synthesis_prompt or _load_synthesis_prompt()

    async def run(
        self,
        query: str,
        catalog: list[CatalogEntry],
        history: Iterable[tuple[str, str]] | None = None,
        pinned_types: list[SourceType] | None = None,
    ) -> AsyncIterator[StudioEvent]:
        by_id = {e.id: e for e in catalog}

        # Stage 1: pick documents
        cls_result = await self._classifier.classify(
            query=query,
            catalog=catalog,
            history=history,
            pinned_types=pinned_types,
        )

        # Resolve selected ids → CatalogEntry objects.
        selected_docs: list[CatalogEntry] = [by_id[sid] for sid in cls_result.selected_doc_ids if sid in by_id]

        # Fallback: if classifier returned nothing usable, use the pin-filtered scope (or whole catalog).
        if not selected_docs:
            if pinned_types:
                selected_docs = [e for e in catalog if e.category in pinned_types]
            else:
                selected_docs = list(catalog)

        types_used = sorted({d.category for d in selected_docs})

        yield StudioEvent(
            "routing_decided",
            {
                "types_used": types_used,
                "is_cross_domain": len(types_used) > 1,
                "classifier_confidence": 0.0 if cls_result.used_fallback else 1.0,
                "reasoning": cls_result.reasoning,
                "used_fallback": cls_result.used_fallback,
                "selected_docs": [{"id": d.id, "name": d.name, "type": d.category} for d in selected_docs],
            },
        )

        # Stage 2: retrieve
        chunks, docs_used = await self._aggregator.aggregate(query, selected_docs)

        for i, c in enumerate(chunks, start=1):
            c.source.setdefault("citation_id", f"c{i}")

        yield StudioEvent(
            "retrieval_done",
            {
                "documents_used": [
                    {"doc_id": d.doc_id, "name": d.name, "type": d.type, "citation_count": d.citation_count}
                    for d in docs_used
                ],
                "chunk_count": len(chunks),
            },
        )

        if not chunks:
            fallback = (
                "I couldn't find relevant material in the selected documents. "
                "Try rephrasing, or pin a different source using the target button in the header."
            )
            yield StudioEvent("token", {"delta": fallback})
            yield StudioEvent("citations_resolved", {"citations": []})
            yield StudioEvent(
                "done",
                {
                    "confidence": "low",
                    "warnings": ["empty-retrieval"],
                    "routing": {
                        "types_used": types_used,
                        "documents_used": [],
                        "primary_document_id": None,
                        "is_cross_domain": len(types_used) > 1,
                        "classifier_confidence": 0.0 if cls_result.used_fallback else 1.0,
                    },
                },
            )
            return

        # Stage 3: synthesize
        user_payload = self._build_synthesis_user_message(query, chunks, history)
        try:
            full_text = await asyncio.to_thread(
                self._llm.call,
                prompt=user_payload,
                system_instruction=self._synthesis_prompt,
                max_tokens=8192,
                add_warning=False,
            )
        except Exception as exc:
            logger.exception("Studio synthesis failed: {}", exc)
            yield StudioEvent("error", {"message": f"Synthesis failed: {exc}"})
            return

        yield StudioEvent("token", {"delta": full_text})

        citations_payload = self._build_citations_payload(full_text, chunks)
        yield StudioEvent("citations_resolved", {"citations": citations_payload})

        primary_doc = self._primary_document(citations_payload)
        yield StudioEvent(
            "done",
            {
                "confidence": self._infer_confidence(citations_payload),
                "warnings": [],
                "routing": {
                    "types_used": types_used,
                    "documents_used": [
                        {"doc_id": d.doc_id, "name": d.name, "type": d.type, "citation_count": d.citation_count}
                        for d in docs_used
                    ],
                    "primary_document_id": primary_doc,
                    "is_cross_domain": len(types_used) > 1,
                    "classifier_confidence": 0.0 if cls_result.used_fallback else 1.0,
                },
            },
        )

    @staticmethod
    def _build_synthesis_user_message(
        query: str,
        chunks: list[AggregatedChunk],
        history: Iterable[tuple[str, str]] | None,
    ) -> str:
        lines: list[str] = []
        if history:
            lines.append("Prior turns (for context):")
            for role, content in list(history)[-3:]:
                lines.append(f"  {role}: {content}")
            lines.append("")
        lines.append(f"Question: {query}")
        lines.append("")
        lines.append("Sections:")
        for c in chunks:
            cid = c.source.get("citation_id")
            section = (
                c.source.get("section")
                or c.source.get("node_id")
                or c.source.get("section_label")
                or "?"
            )
            page = c.source.get("page") or (c.source.get("pages") or [None])[0]
            lines.append(
                f"[id={cid}] ({c.source_type}) {c.doc_name} — § {section}"
                + (f" (p. {page})" if page else "")
            )
            lines.append(c.text.strip())
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_citations_payload(answer_text: str, chunks: list[AggregatedChunk]) -> list[dict]:
        cited_ids = set(m.group(1) for m in re.finditer(r"\[CITE:([^\]]+)\]", answer_text))
        by_id = {c.source.get("citation_id"): c for c in chunks}
        out: list[dict] = []
        for cid in cited_ids:
            c = by_id.get(cid)
            if not c:
                continue
            section = (
                c.source.get("section")
                or c.source.get("node_id")
                or c.source.get("section_label")
                or "?"
            )
            page = c.source.get("page") or (c.source.get("pages") or [None])[0]
            out.append(
                {
                    "id": cid,
                    "section": str(section),
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "type": c.source_type,
                    "page": page,
                    "quote": c.text.strip()[:280],
                    "amended_on": c.source.get("amended_on"),
                }
            )
        return out

    @staticmethod
    def _primary_document(citations: list[dict]) -> str | None:
        if not citations:
            return None
        counts: dict[str, int] = {}
        for c in citations:
            counts[c["doc_id"]] = counts.get(c["doc_id"], 0) + 1
        total = sum(counts.values())
        best_doc, best = max(counts.items(), key=lambda kv: kv[1])
        return best_doc if best / max(total, 1) >= 0.6 else None

    @staticmethod
    def _infer_confidence(citations: list[dict]) -> str:
        if not citations:
            return "low"
        if len(citations) >= 3:
            return "high"
        return "medium"
