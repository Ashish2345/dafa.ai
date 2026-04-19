"""Non-streaming query endpoint.

This module stays deliberately thin: auth, quota, delegate to the orchestrator,
serialize. All business logic lives in ``app.services.query``.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.db.mongodb import get_database
from app.models.schemas import QueryRequest, QueryResponse
from app.services.query.factory import build_query_orchestrator
from app.services.query.quota import QuotaService
from app.utils.auth import get_current_user
from app.utils.exceptions import AppException

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse, summary="Ask a question")
async def query(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Ask a question against ingested documents. Enforces daily search quota."""
    user_id = current_user["sub"]
    quota = QuotaService(db)
    quota.raise_if_exceeded(await quota.check_search(user_id))

    try:
        orchestrator = build_query_orchestrator()
        result = await orchestrator.run_query(request)
        await quota.increment_search(user_id)

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            chunks=orchestrator.retrieval.serialize_chunks(result.chunks),
            sources=result.sources,
            metadata=result.metadata,
        )

    except (AppException, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_QUERY_FAILED",
            message=str(e),
        ) from e
