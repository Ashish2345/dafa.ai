"""Retrieval service — thin wrapper over the strategy factory.

Keeps the endpoint code ignorant of which retrieval strategy is in play and
gives us one place to add features that span strategies (e.g. per-user
filtering, telemetry).
"""

from typing import Any, Dict, List, Optional

from app.services.retrieval.base import RetrievedChunk
from app.services.retrieval.factory import RetrievalFactory


class RetrievalService:
    """Resolves a strategy and runs a single retrieval call."""

    def __init__(self) -> None:
        # Navigator LLM usage from the most recent ``retrieve`` call. Populated
        # from the underlying strategy so the orchestrator can credit the right
        # amount to per-user billing.
        self.last_nav_usage: Dict[str, Any] = {
            "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
            "thinking_tokens": 0, "cost_usd": 0.0,
        }

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        strategy = await RetrievalFactory.get_strategy(strategy_name)
        chunks = await strategy.retrieve(
            query=query,
            top_k=top_k,
            filter_conditions=filter_conditions,
            collection_name=collection_name,
        )
        self.last_nav_usage = dict(getattr(strategy, "last_nav_usage", {}) or self.last_nav_usage)
        return chunks

    @staticmethod
    def unique_document_names(chunks: List[RetrievedChunk]) -> List[str]:
        """Collect document names seen across chunks — used for UI status messages."""
        names: List[str] = []
        seen: set[str] = set()
        for c in chunks:
            name = c.source.get("document_name", "") if isinstance(c.source, dict) else ""
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    @staticmethod
    def serialize_chunks(chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Convert chunks into the JSON shape the frontend expects."""
        return [
            {"text": c.text, "source": c.source, "score": c.score, "metadata": c.metadata}
            for c in chunks
        ]
