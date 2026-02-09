"""
Documents endpoint.

Handles fetching documents from collections and PDF page images.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.services.vector_store import VectorStoreService
from app.utils.exceptions import AppException
from app.utils.security import get_api_key
from loguru import logger
from qdrant_client.http import models

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentInfo(BaseModel):
    """Information about a document."""

    document_id: str = Field(..., description="Document identifier")
    collection_name: str = Field(..., description="Collection name")
    chunk_count: int = Field(default=0, description="Number of chunks for this document")
    metadata: Optional[dict] = Field(default_factory=dict, description="Document metadata")


class DocumentsResponse(BaseModel):
    """Response model for documents list."""

    documents: List[DocumentInfo] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of documents")
    collection_name: str
    status: str = "success"


@router.get("/collection/{collection_name}", response_model=DocumentsResponse, summary="Get documents in a collection")
async def get_collection_documents(
    collection_name: str,
    api_key: str = Depends(get_api_key),
):
    """
    Get all documents in a specific collection.

    Args:
        collection_name: Name of the collection
        api_key: API key for authentication

    Returns:
        DocumentsResponse with list of documents
    """
    logger.info(f"Fetching documents for collection: {collection_name}")

    try:
        vector_store = VectorStoreService()
        all_collections = vector_store.get_all_collections()

        if collection_name not in all_collections:
            raise AppException(
                status_code=status.HTTP_404_NOT_FOUND,
                error_code="E_COLLECTION_NOT_FOUND",
                message=f"Collection '{collection_name}' not found",
            )

        # Get all points from the collection to extract unique document_ids
        document_map = {}
        offset = None
        batch_size = 100

        while True:
            result = vector_store.client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points = result[0]
            next_offset = result[1]

            for point in points:
                doc_id = point.payload.get("document_id", "unknown")
                if doc_id not in document_map:
                    document_map[doc_id] = {
                        "document_id": doc_id,
                        "chunk_count": 0,
                        "metadata": {k: v for k, v in point.payload.items() if k not in ["text", "chunk_id", "document_id"]},
                    }
                document_map[doc_id]["chunk_count"] += 1

            if next_offset is None or len(points) == 0:
                break

            offset = next_offset

        documents = [
            DocumentInfo(
                document_id=doc["document_id"],
                collection_name=collection_name,
                chunk_count=doc["chunk_count"],
                metadata=doc["metadata"],
            )
            for doc in document_map.values()
        ]

        logger.info(f"Found {len(documents)} documents in collection '{collection_name}'")
        return DocumentsResponse(
            documents=documents,
            total=len(documents),
            collection_name=collection_name,
            status="success",
        )

    except AppException:
        raise
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_INTERNAL_SERVER_ERROR",
            message=f"An unexpected error occurred: {str(e)}",
        ) from e
