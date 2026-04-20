"""Run retrieval across a selected list of documents and aggregate chunks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Iterable, Literal

from loguru import logger

from app.services.retrieval.factory import RetrievalFactory
from app.services.studio.classifier import CatalogEntry, SourceType


@dataclass
class AggregatedChunk:
    text: str
    source: dict
    score: float
    doc_id: str
    doc_name: str
    source_type: SourceType
    # Phase 14: 'private' when the originating document is a user's workspace
    # upload (scope='private'); 'public' otherwise. Threaded into the SSE
    # citation payload so the frontend's ActionBar can disable Share when any
    # citation is private.
    source_scope: Literal["public", "private"] = "public"


@dataclass
class DocumentUsage:
    doc_id: str
    name: str
    type: SourceType
    citation_count: int


async def _retrieve_one(collection_name: str, query: str, top_k: int):
    """Run PageIndexStrategy retrieval for a single collection.

    PageIndexStrategy expects `collection_name` as a case-insensitive substring
    of the document title. We pass the document's title here.
    """
    try:
        strategy = await RetrievalFactory.get_strategy(strategy_name="page_index")
        return await strategy.retrieve(query=query, top_k=top_k, collection_name=collection_name)
    except Exception as exc:
        logger.warning("Studio retrieval failed for {}: {}", collection_name, exc)
        return []


class DocumentAggregator:
    """Aggregates retrieval results across a pre-selected list of documents."""

    async def aggregate(
        self,
        query: str,
        selected_docs: Iterable[CatalogEntry],
        per_doc_k: int = 5,
        top_k: int = 8,
    ) -> tuple[list[AggregatedChunk], list[DocumentUsage]]:
        selected = list(selected_docs)
        if not selected:
            return [], []

        tasks = [
            asyncio.create_task(_retrieve_one(d.name, query, per_doc_k))
            for d in selected
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        chunks: list[AggregatedChunk] = []
        for doc, returned in zip(selected, results):
            for r in returned or []:
                if isinstance(r, dict):
                    text = r.get("text", "") or ""
                    source = r.get("source", {}) or {}
                    score = float(r.get("score", 0) or 0)
                else:
                    text = getattr(r, "text", "") or ""
                    source = getattr(r, "source", None) or {}
                    score = float(getattr(r, "score", 0) or 0)
                chunks.append(
                    AggregatedChunk(
                        text=text,
                        source=source,
                        score=score,
                        doc_id=doc.id,
                        doc_name=doc.name,
                        source_type=doc.category,
                        source_scope=getattr(doc, "scope", "public") or "public",
                    )
                )

        chunks.sort(key=lambda c: c.score, reverse=True)
        top = chunks[:top_k]

        by_doc: dict[str, DocumentUsage] = {}
        for c in top:
            existing = by_doc.get(c.doc_id)
            if existing:
                existing.citation_count += 1
            else:
                by_doc[c.doc_id] = DocumentUsage(
                    doc_id=c.doc_id, name=c.doc_name, type=c.source_type, citation_count=1
                )
        return top, list(by_doc.values())


# Keep the old name as an alias for backward-compatible imports.
CrossTypeAggregator = DocumentAggregator
