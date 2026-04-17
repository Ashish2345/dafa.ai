"""
Streaming query endpoint — SSE progress events + final answer.

Breaks the query flow into granular steps so the frontend shows
real-time progress as each phase completes.
"""

import asyncio
import json
import re
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.config.plan_loader import plan_catalog
from app.db.mongodb import get_database
from app.db.repositories.usage_repository import UsageRepository
from app.db.repositories.user_repository import UserRepository
from app.services.llm import LLMService
from app.services.retrieval.factory import RetrievalFactory
from app.utils.auth import get_current_user

router = APIRouter(prefix="/query", tags=["query"])


class _ConversationMessage(BaseModel):
    role: str
    content: str


class StreamQueryRequest(BaseModel):
    query: str = Field(..., description="User query/question", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = Field(default=True)
    strategy: Optional[str] = Field(default=None)
    collection_name: Optional[str] = Field(default=None)
    filter_conditions: Optional[Dict[str, Any]] = Field(default=None)
    response_language: Optional[str] = Field(
        default=None,
        description="Response language: 'en' or 'ne'. When omitted, auto-detected from document content.",
    )
    conversation_history: Optional[list[_ConversationMessage]] = Field(
        default=None,
        description="Last few messages for follow-up context.",
    )


_DEVANAGARI = re.compile(r'[\u0900-\u097F]')


def _detect_language(chunks: list) -> str:
    """Infer document language from chunk text — Devanagari content → 'ne'."""
    for c in chunks[:3]:
        text = c.text if hasattr(c, 'text') else c.get('text', '')
        if _DEVANAGARI.search(text):
            return 'ne'
    return 'en'


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream", summary="Ask a question (streaming progress)")
async def query_stream(
    request: StreamQueryRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Streaming query with real-time progress events via SSE."""
    user_id = current_user["sub"]

    # ─── Plan quota enforcement ──────────────────────────────────────
    user_doc = await UserRepository(db).get_by_id(user_id)
    plan_id = (user_doc or {}).get("plan") or plan_catalog.default_plan_id
    plan = plan_catalog.get(plan_id)

    usage_repo = UsageRepository(db)
    allowed, used, remaining = await usage_repo.check_quota(
        user_id, "search", plan.limits.searches_per_day
    )

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": "plan_limit_reached",
                "action": "search",
                "plan_id": plan_id,
                "plan_name": plan.name,
                "limit": plan.limits.searches_per_day,
                "used": used,
                "message": (
                    f"You've used all {plan.limits.searches_per_day} questions on your "
                    f"{plan.name} plan today. Upgrade to continue asking."
                ),
            },
        )

    async def event_generator():
        try:
            # Step 1: Opening the document
            yield _sse_event("status", {
                "step": "loading",
                "message": "Opening the document…",
            })

            strategy = await RetrievalFactory.get_strategy(request.strategy)

            # Step 2: Searching for relevant sections
            yield _sse_event("status", {
                "step": "navigating",
                "message": "Searching for relevant sections…",
            })

            chunks = await strategy.retrieve(
                query=request.query,
                top_k=request.top_k,
                filter_conditions=request.filter_conditions,
                collection_name=request.collection_name,
            )

            if not chunks:
                yield _sse_event("status", {
                    "step": "extracting",
                    "message": "No relevant sections found",
                })
            else:
                # Collect document names for better status
                doc_names = set()
                for c in chunks:
                    name = c.source.get("document_name", "")
                    if name:
                        doc_names.add(name)
                doc_label = next(iter(doc_names), "document") if doc_names else "document"

                yield _sse_event("status", {
                    "step": "extracting",
                    "message": f"Reading {len(chunks)} sections from {doc_label}…",
                })

            # Step 3: Synthesize answer
            answer = ""
            synthesis_meta = {}
            if request.use_llm and chunks:
                yield _sse_event("status", {
                    "step": "synthesizing",
                    "message": "Writing your answer…",
                })

                llm = LLMService()
                # User picks response language; fall back to auto-detect from content
                language = request.response_language or _detect_language(chunks)

                # Prepend conversation context for follow-up questions
                query_for_llm = request.query
                if request.conversation_history:
                    ctx_lines = []
                    for msg in request.conversation_history:
                        label = "User" if msg.role == "user" else "Assistant"
                        ctx_lines.append(f"[{label}: {msg.content}]")
                    query_for_llm = "\n".join(ctx_lines) + f"\n\nFollow-up question: {request.query}"
                # Run synthesis in thread so SSE events can flush
                answer, synthesis_meta = await asyncio.to_thread(
                    llm.synthesize, query_for_llm, chunks, language,
                )

            # Track total LLM cost on the daily usage counter (no per-call records)
            total_cost = (synthesis_meta.get("cost_usd") or 0.0) if synthesis_meta else 0.0
            total_tokens = (
                (synthesis_meta.get("input_tokens") or 0)
                + (synthesis_meta.get("output_tokens") or 0)
                + (synthesis_meta.get("thinking_tokens") or 0)
            ) if synthesis_meta else 0

            # Generate contextual follow-up suggestions (best-effort, non-blocking)
            follow_ups: list[str] = []
            if answer and chunks:
                try:
                    fu_llm = LLMService(
                        model="gemini-2.0-flash",
                        temperature=0.4,
                        max_tokens=256,
                    )
                    fu_prompt = (
                        "You are a Nepali legal research assistant. "
                        "Given this Q&A, suggest exactly 3 short follow-up questions "
                        "a finance professional would naturally ask next.\n\n"
                        f"Question: {request.query}\n"
                        f"Answer (excerpt): {answer[:600]}\n\n"
                        "Return a JSON array of 3 strings, each under 60 characters. "
                        "No markdown, no explanation — just the JSON array."
                    )
                    raw, fu_meta = await asyncio.to_thread(
                        fu_llm.call,
                        fu_prompt,
                        add_warning=False,
                        response_mime_type="application/json",
                        return_metadata=True,
                    )
                    parsed = json.loads(raw if isinstance(raw, str) else raw[0])
                    if isinstance(parsed, list):
                        follow_ups = [s for s in parsed if isinstance(s, str)][:3]
                    total_cost += fu_meta.get("cost_usd") or 0.0
                    total_tokens += (fu_meta.get("input_tokens") or 0) + (fu_meta.get("output_tokens") or 0)
                except Exception as e:
                    logger.debug(f"Follow-up generation skipped: {e}")

            # Increment daily LLM cost + tokens on the existing usage counter
            if total_cost > 0 or total_tokens > 0:
                await usage_repo.increment_float(user_id, "llm_cost_usd", total_cost)
                await usage_repo.increment(user_id, "llm_tokens", amount=total_tokens)

            # Build response
            sources = [chunk.source for chunk in chunks]
            response_data = {
                "query": request.query,
                "answer": answer,
                "chunks": [
                    {"text": c.text, "source": c.source, "score": c.score, "metadata": c.metadata}
                    for c in chunks
                ],
                "sources": sources,
                "follow_ups": follow_ups,
                "metadata": {
                    "strategy": request.strategy or "default",
                    "chunks_retrieved": len(chunks),
                },
            }

            # Count against quota
            await usage_repo.increment(user_id, "search")

            yield _sse_event("done", response_data)

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield _sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
