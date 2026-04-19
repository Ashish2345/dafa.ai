"""Query orchestrator — composes retrieval, synthesis, and follow-up services.

Two top-level entry points:

  * :meth:`run_query`   — non-streaming, returns a full :class:`QueryResult`.
                          Used by ``POST /query``.
  * :meth:`stream_query` — async generator yielding pre-formatted SSE frames.
                           Used by ``POST /query/stream``.

Both share the same underlying services so behaviour stays consistent.
"""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from app.models.schemas import QueryRequest, StreamQueryRequest
from app.services.query.followups import FollowupService
from app.services.query.quota import QuotaService
from app.services.query.retrieval import RetrievalService
from app.services.query.synthesis import SynthesisService
from app.services.retrieval.base import RetrievedChunk
from app.utils.language import detect_language, detect_language_from_text
from app.utils.sse import format_event, is_rate_limit_error, rate_limit_event


_STATUS_MESSAGES: Dict[str, Dict[str, str]] = {
    "loading":      {"en": "Getting your document ready…",          "ne": "तपाईंको कागजात खोल्दै…"},
    "navigating":   {"en": "Finding the sections you need…",        "ne": "तपाईंको प्रश्नसँग मिल्ने दफा खोज्दै…"},
    "no_results":   {"en": "No matching sections found",            "ne": "मिल्दो दफा भेटिएन"},
    "extracting":   {"en": "Reading {n} relevant sections from {doc}…", "ne": "{doc} का {n} दफा अध्ययन गर्दै…"},
    "synthesizing": {"en": "Preparing your answer…",                "ne": "तपाईंको उत्तर तयार पार्दै…"},
}


def _status_text(key: str, lang: str, **kwargs: Any) -> str:
    template = _STATUS_MESSAGES[key].get(lang) or _STATUS_MESSAGES[key]["en"]
    return template.format(**kwargs) if kwargs else template


@dataclass
class QueryResult:
    """Non-streaming query response payload."""

    query: str
    answer: str
    chunks: List[RetrievedChunk]
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryOrchestrator:
    """Front door for the query pipeline."""

    def __init__(
        self,
        retrieval: Optional[RetrievalService] = None,
        synthesis: Optional[SynthesisService] = None,
        followups: Optional[FollowupService] = None,
    ):
        self.retrieval = retrieval or RetrievalService()
        self.synthesis = synthesis or SynthesisService()
        self.followups = followups or FollowupService()

    # ------------------------------------------------------------------
    # Non-streaming flow
    # ------------------------------------------------------------------

    async def run_query(self, req: QueryRequest) -> QueryResult:
        logger.info(f"Query: {req.query[:100]!r}, strategy={req.strategy or 'default'}")

        chunks = await self.retrieval.retrieve(
            query=req.query,
            top_k=req.top_k,
            strategy_name=req.strategy,
            collection_name=req.collection_name,
            filter_conditions=req.filter_conditions,
        )

        answer = ""
        if req.use_llm and chunks:
            language = detect_language(chunks)
            outcome = await self.synthesis.synthesize_blocking(
                query=req.query, chunks=chunks, language=language,
            )
            answer = outcome.answer

        return QueryResult(
            query=req.query,
            answer=answer,
            chunks=chunks,
            sources=[c.source for c in chunks],
            metadata={
                "strategy": req.strategy or "default",
                "chunks_retrieved": len(chunks),
            },
        )

    # ------------------------------------------------------------------
    # Streaming flow
    # ------------------------------------------------------------------

    async def stream_query(
        self, req: StreamQueryRequest, user_id: str, quota: QuotaService,
    ) -> AsyncIterator[str]:
        """Yield SSE frames as the query progresses.

        Event sequence:
          status → (status)* → (token)* → done → [follow_ups]  ← happy path
          status → … → error                                   ← terminal error
        """
        try:
            status_lang = req.response_language or detect_language_from_text(req.query)

            yield format_event("status", {"step": "loading", "message": _status_text("loading", status_lang)})

            chunks = await self.retrieval.retrieve(
                query=req.query,
                top_k=req.top_k,
                strategy_name=req.strategy,
                collection_name=req.collection_name,
                filter_conditions=req.filter_conditions,
            )

            yield format_event("status", {"step": "navigating", "message": _status_text("navigating", status_lang)})

            if not chunks:
                yield format_event("status", {"step": "extracting", "message": _status_text("no_results", status_lang)})
            else:
                doc_names = self.retrieval.unique_document_names(chunks)
                doc_label = doc_names[0] if doc_names else ("कागजात" if status_lang == "ne" else "document")
                yield format_event("status", {
                    "step": "extracting",
                    "message": _status_text("extracting", status_lang, n=len(chunks), doc=doc_label),
                })

            answer = ""
            synthesis_meta: Dict[str, Any] = {}

            if req.use_llm and chunks:
                yield format_event("status", {"step": "synthesizing", "message": _status_text("synthesizing", status_lang)})

                language = req.response_language or detect_language(chunks)

                async for text in self.synthesis.stream_tokens(
                    query=req.query,
                    chunks=chunks,
                    language=language,
                    history=req.conversation_history,
                ):
                    yield format_event("token", {"text": text})

                outcome = self.synthesis.last_outcome
                if outcome.error_event:
                    if outcome.error_event.get("code") in ("rate_limit", "api_error", "quota_exceeded"):
                        yield rate_limit_event(detail=outcome.error_event.get("message", ""))
                    else:
                        yield format_event("error", outcome.error_event)
                    return

                answer = outcome.answer
                synthesis_meta = outcome.metadata

            # Always emit `done` so the frontend can finalise rendering —
            # follow-ups are generated AFTER this event and streamed
            # separately so they never block the answer.
            await quota.increment_search(user_id)

            yield format_event("done", {
                "query": req.query,
                "answer": answer,
                "chunks": self.retrieval.serialize_chunks(chunks),
                "sources": [c.source for c in chunks],
                "follow_ups": [],
                "metadata": {
                    "strategy": req.strategy or "default",
                    "chunks_retrieved": len(chunks),
                },
            })

            # Best-effort follow-ups off the critical path.
            fu = await self.followups.generate(query=req.query, answer=answer, chunks=chunks)
            if fu.suggestions:
                yield format_event("follow_ups", {"follow_ups": fu.suggestions})

            # Aggregate LLM spend for the daily counter.
            total_cost = (synthesis_meta.get("cost_usd") or 0.0) + fu.cost_usd
            total_tokens = (
                (synthesis_meta.get("input_tokens") or 0)
                + (synthesis_meta.get("output_tokens") or 0)
                + (synthesis_meta.get("thinking_tokens") or 0)
                + fu.tokens
            )
            logger.info(
                f"Query done | chunks={len(chunks)} | tokens={total_tokens} | "
                f"cost=${total_cost:.6f} | follow_ups={len(fu.suggestions)}"
            )
            await quota.increment_llm_usage(user_id, total_cost, total_tokens)

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            msg = str(e)
            if is_rate_limit_error(msg):
                yield rate_limit_event(detail=msg)
            else:
                yield format_event("error", {"message": msg})
