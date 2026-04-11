"""
Query orchestrator — retrieves chunks via strategy, synthesizes via LLM.

Replaces the 647-line RAGOrchestrator with ~60 lines.
"""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.services.llm import LLMService
from app.services.retrieval.base import RetrievedChunk
from app.services.retrieval.factory import RetrievalFactory


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
            answer = self.llm.synthesize(user_query, chunks)

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
