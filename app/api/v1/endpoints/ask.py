"""
RAG Query endpoint.

Handles user queries against the ingested finance acts.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query, status

from app.services.rag import RAGOrchestrator
from app.utils.exceptions import AppException
from app.utils.auth import get_current_user
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/ask", tags=["rag"])


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(..., description="User query/question", min_length=1)
    top_k: int = Field(default=5, description="Number of top chunks to retrieve", ge=1, le=20)
    use_llm: bool = Field(default=True, description="Whether to use LLM for answer synthesis")
    filter_conditions: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional filters (e.g., {'act_name': 'VAT Act'})"
    )
    collection_name: Optional[str] = Field(
        default=None, description="Optional specific collection to search (overrides collection routing)"
    )


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    query: str
    answer: str
    chunks: list[Dict[str, Any]] = Field(default_factory=list)
    sources: list[Dict[str, Any]] = Field(default_factory=list)
    status: str = "success"
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Response metadata (processing time, tokens, etc.)"
    )
    warnings: Optional[list[str]] = Field(
        default_factory=list,
        description="Any warnings about the response (e.g., truncation)"
    )


@router.post("", response_model=QueryResponse, summary="Ask a question about finance acts")
async def ask_question(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Ask a question about finance acts using RAG.

    This endpoint:
    1. Generates embedding for the query
    2. Searches Qdrant for relevant chunks
    3. Synthesizes answer using Gemini 2.5 Flash

    Args:
        request: Query request with question and options
        api_key: API key for authentication

    Returns:
        QueryResponse with answer and sources
    """
    logger.info(f"Processing query: {request.query[:100]}...")

    try:
        from app.settings import settings as app_settings
        orchestrator = RAGOrchestrator(top_k=request.top_k)

        # Route to PageIndex (vectorless) or traditional vector RAG
        if app_settings.use_page_index:
            from app.db.mongodb import get_database
            db = await get_database()
            result = await orchestrator.async_query(
                user_query=request.query,
                filter_conditions=request.filter_conditions,
                use_llm=request.use_llm,
                collection_name=request.collection_name,
                db=db,
            )
        else:
            result = orchestrator.query(
                user_query=request.query,
                filter_conditions=request.filter_conditions,
                use_llm=request.use_llm,
                collection_name=request.collection_name,
            )

        # Check for errors
        if "error" in result:
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code="E_RAG_ERROR",
                message=result.get("error", "Error processing query"),
            )

        # Extract metadata and warnings
        metadata = result.get("metadata", {})
        warnings = []
        
        # Check for truncation or incomplete responses
        answer_metadata = metadata.get("answer_metadata", {})
        if answer_metadata.get("is_truncated"):
            warnings.append("Response was truncated due to length limits. Some information may be incomplete.")
        elif not answer_metadata.get("is_complete", True):
            warnings.append("Response may be incomplete. Consider refining your query for more specific information.")
        
        # Check if no chunks were found
        if not result.get("chunks"):
            warnings.append("No relevant information found in the knowledge base. The answer may be incomplete.")
        
        # Build response
        response = QueryResponse(
            query=result["query"],
            answer=result.get("answer", ""),
            chunks=result.get("chunks", []),
            sources=result.get("sources", []),
            status="success",
            metadata=metadata,
            warnings=warnings if warnings else None,
        )
        
        return response

    except AppException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_INTERNAL_SERVER_ERROR",
            message=f"An unexpected error occurred: {str(e)}",
        ) from e


@router.get("", response_model=QueryResponse, summary="Ask a question (GET)")
async def ask_question_get(
    q: str = Query(..., description="User query/question", min_length=1),
    top_k: int = Query(default=5, description="Number of top chunks", ge=1, le=20),
    use_llm: bool = Query(default=True, description="Use LLM for synthesis"),
    current_user: dict = Depends(get_current_user),
):
    """
    Ask a question using GET method (convenience endpoint).

    Args:
        q: User query
        top_k: Number of top chunks
        use_llm: Use LLM for synthesis
        api_key: API key

    Returns:
        QueryResponse
    """
    request = QueryRequest(query=q, top_k=top_k, use_llm=use_llm)
    return await ask_question(request, api_key)
