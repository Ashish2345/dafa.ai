"""
Query orchestrator — retrieves chunks via strategy, synthesizes via LLM.

Replaces the 647-line RAGOrchestrator with ~60 lines.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.services.llm import LLMService
from app.services.retrieval.base import RetrievedChunk
from app.services.retrieval.factory import RetrievalFactory

_DEVANAGARI = re.compile(r'[\u0900-\u097F]')


def _detect_language(chunks: list) -> str:
    """Infer document language from chunk text — Devanagari content → 'ne'."""
    for c in chunks[:3]:
        text = c.text if hasattr(c, 'text') else c.get('text', '')
        if _DEVANAGARI.search(text):
            return 'ne'
    return 'en'


@dataclass
class QueryResult:
    """Result returned by QueryOrchestrator.query()."""

    query: str
    answer: str
    chunks: list[RetrievedChunk]
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryOrchestrator:
    """Thin orchestrator: factory -> retrieve -> LLM synthesize."""

    def __init__(self, llm_service: LLMService | None = None):
        self.llm = llm_service or LLMService()

    async def query(
        self,
        user_query: str,
        top_k: int = 5,
        strategy_name: str | None = None,
        collection_name: str | None = None,
        filter_conditions: dict[str, Any] | None = None,
        use_llm: bool = True,
    ) -> QueryResult:
        logger.info(f"Query: {user_query[:100]!r}, strategy={strategy_name or 'default'}")

        strategy = await RetrievalFactory.get_strategy(strategy_name)

        chunks = await strategy.retrieve(
            query=user_query,
            top_k=top_k,
            filter_conditions=filter_conditions,
            collection_name=collection_name,
        )

        answer = ""
        if use_llm and chunks:
            doc_language = _detect_language(chunks)
            answer = self.llm.synthesize(user_query, chunks, doc_language)

        sources = [chunk.source for chunk in chunks]

        return QueryResult(
            query=user_query,
            answer=answer,
            chunks=chunks,
            sources=sources,
            metadata={
                "strategy": strategy_name or "default",
                "chunks_retrieved": len(chunks),
            },
        )
