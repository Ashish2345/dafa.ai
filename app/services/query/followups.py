"""Follow-up question suggestions — best-effort, always off the critical path."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.llm import LLMService
from app.services.retrieval.base import RetrievedChunk


_FOLLOWUP_PROMPT = (
    "You are a Nepali legal research assistant. "
    "Given this Q&A, suggest exactly 3 short follow-up questions "
    "a finance professional would naturally ask next.\n\n"
    "Question: {query}\n"
    "Answer (excerpt): {answer}\n\n"
    "Return a JSON array of 3 strings, each under 60 characters. "
    "No markdown, no explanation — just the JSON array."
)


@dataclass
class FollowupsResult:
    """Suggestions plus aggregated cost/token accounting."""

    suggestions: List[str] = field(default_factory=list)
    cost_usd: float = 0.0
    tokens: int = 0


class FollowupService:
    """Generates short follow-up questions from a completed answer.

    Failures (429, parse errors, model hiccups) are swallowed — this is a
    "nice to have" surface and must never affect the main answer path.
    """

    def __init__(self, llm: Optional[LLMService] = None):
        # ``max_tokens`` is the total budget Gemini 2.5 reserves for the
        # response including any internal thinking; 256 was enough for
        # 2.0-flash but gets swallowed here before the JSON array is emitted.
        # 1024 is plenty for a 3-item array of short strings.
        self._llm = llm or LLMService(temperature=0.4, max_tokens=1024)

    async def generate(
        self, query: str, answer: str, chunks: List[RetrievedChunk],
    ) -> FollowupsResult:
        if not (answer and chunks):
            logger.info(
                f"Follow-ups skipped: answer_empty={not answer}, "
                f"chunks_empty={not chunks}"
            )
            return FollowupsResult()

        prompt = _FOLLOWUP_PROMPT.format(query=query, answer=answer[:600])

        try:
            raw, meta = await self._llm.a_call(
                prompt=prompt,
                response_mime_type="application/json",
                return_metadata=True,
            )
        except Exception as e:
            logger.warning(f"Follow-ups LLM call failed: {e}")
            return FollowupsResult()

        try:
            parsed = json.loads(raw if isinstance(raw, str) else str(raw))
        except json.JSONDecodeError as e:
            logger.warning(
                f"Follow-ups JSON parse failed: {e}; raw={raw!r}"
            )
            return FollowupsResult()

        suggestions = (
            [s for s in parsed if isinstance(s, str)][:3]
            if isinstance(parsed, list)
            else []
        )
        logger.info(f"Follow-ups generated: {len(suggestions)} suggestions")
        return FollowupsResult(
            suggestions=suggestions,
            cost_usd=float(meta.get("cost_usd", 0.0) or 0.0),
            tokens=int(meta.get("input_tokens", 0) or 0) + int(meta.get("output_tokens", 0) or 0),
        )
