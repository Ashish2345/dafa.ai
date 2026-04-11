"""
Query endpoint — ask questions against ingested documents.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, status
from loguru import logger
from pydantic import BaseModel, Field

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
):
    """Ask a question against ingested documents."""
    logger.info(f"Query: {request.query[:100]!r}")

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
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_QUERY_FAILED",
            message=str(e),
        ) from e
