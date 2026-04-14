"""
Streaming query endpoint — SSE progress events + final answer.

Sends real-time status updates to the frontend as the query progresses
through tree navigation, section extraction, and answer synthesis.

Event format (Server-Sent Events):
  event: status
  data: {"step": "navigating", "message": "Finding relevant sections..."}

  event: status
  data: {"step": "extracting", "message": "Reading 5 sections..."}

  event: status
  data: {"step": "synthesizing", "message": "Generating answer..."}

  event: done
  data: {full QueryResponse JSON}

  event: error
  data: {"message": "..."}
"""

import asyncio
import json
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
from app.services.retrieval.base import RetrievedChunk
from app.services.retrieval.factory import RetrievalFactory
from app.utils.auth import get_current_user
from app.utils.exceptions import AppException

router = APIRouter(prefix="/query", tags=["query"])


class StreamQueryRequest(BaseModel):
    query: str = Field(..., description="User query/question", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = Field(default=True)
    strategy: Optional[str] = Field(default=None)
    collection_name: Optional[str] = Field(default=None)
    filter_conditions: Optional[Dict[str, Any]] = Field(default=None)


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
            # Step 1: Navigating document tree
            yield _sse_event("status", {
                "step": "navigating",
                "message": "Searching documents...",
            })

            strategy = await RetrievalFactory.get_strategy(request.strategy)
            chunks = await strategy.retrieve(
                query=request.query,
                top_k=request.top_k,
                filter_conditions=request.filter_conditions,
                collection_name=request.collection_name,
            )

            yield _sse_event("status", {
                "step": "extracting",
                "message": f"Found {len(chunks)} relevant sections",
            })

            # Step 2: Synthesize answer
            answer = ""
            if request.use_llm and chunks:
                yield _sse_event("status", {
                    "step": "synthesizing",
                    "message": "Generating answer...",
                })

                llm = LLMService()
                answer = llm.synthesize(request.query, chunks)

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
