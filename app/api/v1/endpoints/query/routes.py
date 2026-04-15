"""
Query endpoint — ask questions against ingested documents.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from app.config.plan_loader import plan_catalog
from app.db.mongodb import get_database
from app.db.repositories.usage_repository import UsageRepository
from app.db.repositories.user_repository import UserRepository
from app.services.query.orchestrator import QueryOrchestrator
from app.utils.auth import get_current_user
from app.utils.exceptions import AppException

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query/question", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = Field(default=True)
    strategy: Optional[str] = Field(default=None, description="'page_index' or 'vector'")
    collection_name: Optional[str] = Field(default=None)
    filter_conditions: Optional[Dict[str, Any]] = Field(default=None)


class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks: list[Dict[str, Any]] = Field(default_factory=list)
    sources: list[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@router.post("", response_model=QueryResponse, summary="Ask a question")
async def query(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Ask a question against ingested documents. Enforces per-plan daily search quota."""
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
        logger.info(f"Quota exceeded for user {user_id} on plan {plan_id} ({used}/{plan.limits.searches_per_day})")
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

    logger.info(f"Query: {request.query[:100]!r} (plan {plan_id}, remaining: {remaining})")

    try:
        orchestrator = QueryOrchestrator()
        result = await orchestrator.query(
            user_query=request.query,
            top_k=request.top_k,
            strategy_name=request.strategy,
            collection_name=request.collection_name,
            filter_conditions=request.filter_conditions,
            use_llm=request.use_llm,
        )

        # Only count successful queries against the quota
        await usage_repo.increment(user_id, "search")

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            chunks=[
                {"text": c.text, "source": c.source, "score": c.score, "metadata": c.metadata}
                for c in result.chunks
            ],
            sources=result.sources,
            metadata=result.metadata,
        )

    except AppException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_QUERY_FAILED",
            message=str(e),
        ) from e
