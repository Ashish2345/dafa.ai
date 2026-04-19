"""Synthesis service — turns retrieved chunks + query into a rendered answer.

Encapsulates:
  * Model choice (swap here when cost / quality trade-offs change).
  * Streaming via Gemini's server-sent chunks.
  * Blocking fallback when streaming fails mid-flight (SDK parse errors,
    transient network issues).
  * Conversation-history prepending for follow-up questions.

The caller owns quota + SSE plumbing; this service just produces text.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from loguru import logger

from app.models.schemas import ConversationMessage
from app.services.llm import LLMService
from app.services.retrieval.base import RetrievedChunk


_STREAM_SENTINEL = object()


@dataclass
class SynthesisOutcome:
    """Result of a synthesis run (streaming or blocking)."""

    answer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_event: Optional[Dict[str, Any]] = None  # set iff the call hit a hard error


class SynthesisService:
    """LLM-backed answer synthesis for retrieved chunks."""

    def __init__(self, llm: Optional[LLMService] = None):
        self._llm = llm or LLMService()

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _prepend_conversation(
        query: str, history: Optional[Iterable[ConversationMessage]]
    ) -> str:
        """Tuck a short transcript in front of the live question, if any."""
        if not history:
            return query
        lines: List[str] = []
        for msg in history:
            label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"[{label}: {msg.content}]")
        return "\n".join(lines) + f"\n\nFollow-up question: {query}"

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_tokens(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        language: str,
        history: Optional[Iterable[ConversationMessage]] = None,
    ) -> AsyncIterator[str]:
        """Yield incremental answer text; the caller wraps each in an SSE event.

        On stream failure, the service falls back to a blocking synthesize
        and yields the full answer as a single chunk. If even the blocking
        call fails, the aggregated result surfaces via :meth:`last_outcome`
        — so callers should read that after exhausting the iterator.
        """
        self._last_outcome = SynthesisOutcome()
        query_for_llm = self._prepend_conversation(query, history)

        stream_gen = self._llm.stream_synthesize(query_for_llm, chunks, language)
        streamed_parts: List[str] = []
        stream_failed = False

        try:
            while True:
                item = await asyncio.to_thread(next, stream_gen, _STREAM_SENTINEL)
                if item is _STREAM_SENTINEL:
                    break
                if isinstance(item, dict) and item.get("__done__"):
                    self._last_outcome.answer = item.get("full_text", "") or "".join(streamed_parts)
                    self._last_outcome.metadata = item.get("metadata", {}) or {}
                    return
                if isinstance(item, str) and item:
                    streamed_parts.append(item)
                    yield item
        except Exception as e:
            logger.warning(
                f"Streaming synthesis failed, falling back to blocking call: {e}"
            )
            stream_failed = True

        # Fallback: blocking synthesize so the user still gets an answer.
        if stream_failed or not streamed_parts:
            answer, metadata = await asyncio.to_thread(
                self._llm.synthesize, query_for_llm, chunks, language,
            )
            if isinstance(metadata, dict) and metadata.get("is_error"):
                # Hard error — stash it so the caller can emit a clean SSE event.
                self._last_outcome.error_event = {
                    "code": metadata.get("error_type", "unknown"),
                    "message": answer,
                }
                return
            self._last_outcome.answer = answer
            self._last_outcome.metadata = metadata or {}
            if answer:
                yield answer
        else:
            # Stream finished mid-way without a __done__ marker — rare, but
            # reconstruct the answer from what we did stream.
            self._last_outcome.answer = "".join(streamed_parts)

    @property
    def last_outcome(self) -> SynthesisOutcome:
        """Outcome metadata from the most recent :meth:`stream_tokens` call."""
        return getattr(self, "_last_outcome", SynthesisOutcome())

    # ------------------------------------------------------------------
    # Blocking
    # ------------------------------------------------------------------

    async def synthesize_blocking(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        language: str,
        history: Optional[Iterable[ConversationMessage]] = None,
    ) -> SynthesisOutcome:
        """Non-streaming synthesis — used by the plain POST /query endpoint."""
        query_for_llm = self._prepend_conversation(query, history)
        answer, metadata = await asyncio.to_thread(
            self._llm.synthesize, query_for_llm, chunks, language,
        )
        outcome = SynthesisOutcome(answer=answer, metadata=metadata or {})
        if isinstance(metadata, dict) and metadata.get("is_error"):
            outcome.error_event = {
                "code": metadata.get("error_type", "unknown"),
                "message": answer,
            }
        return outcome
