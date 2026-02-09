"""
Collections endpoint.

Handles fetching collections (featured notebooks) from Qdrant.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from app.services.rag.collection_router import CollectionRouter
from app.services.vector_store import VectorStoreService
from app.utils.exceptions import AppException
from app.utils.security import get_api_key
from loguru import logger

router = APIRouter(prefix="/collections", tags=["collections"])


class CollectionInfo(BaseModel):
    """Information about a collection."""

    name: str = Field(..., description="Collection name")
    act_name: Optional[str] = Field(None, description="Human-readable act name")
    count: Optional[int] = Field(None, description="Number of points/chunks in collection")
    metadata: Optional[dict] = Field(default_factory=dict, description="Collection metadata")


class CollectionsResponse(BaseModel):
    """Response model for collections list."""

    collections: List[CollectionInfo] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of collections")
    status: str = "success"


class CollectionDetailsResponse(BaseModel):
    """Response model for collection details."""

    name: str
    count: int = Field(default=0, description="Number of points/chunks")
    metadata: Optional[dict] = Field(default_factory=dict)
    status: str = "success"


@router.get("", response_model=CollectionsResponse, summary="Get all collections")
async def get_collections(
    api_key: str = Depends(get_api_key),
):
    """
    Get all collections (featured notebooks) from Qdrant.

    Returns a list of all available collections with their basic information.

    Args:
        api_key: API key for authentication

    Returns:
        CollectionsResponse with list of collections
    """
    logger.info("Fetching all collections from Qdrant")
    try:
        vector_store = VectorStoreService()
        collection_names = vector_store.get_all_collections()

        if not collection_names:
            logger.warning("No collections found in Qdrant")
            return CollectionsResponse(
                collections=[],
                total=0,
                status="success",
            )

        # Get collection info with counts
        collections_info = []
        for collection_name in collection_names:
            try:
                # Get count of points in collection
                # Use scroll to count (more efficient than retrieving all)
                count = 0
                offset = None
                batch_size = 100

                while True:
                    result = vector_store.client.scroll(
                        collection_name=collection_name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False,
                    )
                    points = result[0]
                    next_offset = result[1]

                    count += len(points)

                    if next_offset is None or len(points) == 0:
                        break

                    offset = next_offset

                # Convert collection name to act name
                collection_router = CollectionRouter()
                act_name = collection_router._collection_to_act_name(collection_name)
                
                collections_info.append(
                    CollectionInfo(
                        name=collection_name,
                        act_name=act_name,
                        count=count,
                        metadata={},
                    )
                )
            except Exception as e:
                logger.warning(f"Error getting info for collection '{collection_name}': {e}")
                # Still include the collection even if we can't get count
                collection_router = CollectionRouter()
                act_name = collection_router._collection_to_act_name(collection_name)
                
                collections_info.append(
                    CollectionInfo(
                        name=collection_name,
                        act_name=act_name,
                        count=None,
                        metadata={},
                    )
                )

        logger.info(f"Found {len(collections_info)} collections")
        return CollectionsResponse(
            collections=collections_info,
            total=len(collections_info),
            status="success",
        )

    except Exception as e:
        logger.error(f"Error fetching collections: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_INTERNAL_SERVER_ERROR",
            message=f"An unexpected error occurred: {str(e)}",
        ) from e


@router.get("/{collection_name}", response_model=CollectionDetailsResponse, summary="Get collection details")
async def get_collection_details(
    collection_name: str,
    api_key: str = Depends(get_api_key),
):
    """
    Get detailed information about a specific collection.

    Args:
        collection_name: Name of the collection
        api_key: API key for authentication

    Returns:
        CollectionDetailsResponse with collection details
    """
    logger.info(f"Fetching details for collection: {collection_name}")

    try:
        vector_store = VectorStoreService()
        all_collections = vector_store.get_all_collections()

        if collection_name not in all_collections:
            raise AppException(
                status_code=status.HTTP_404_NOT_FOUND,
                error_code="E_COLLECTION_NOT_FOUND",
                message=f"Collection '{collection_name}' not found",
            )

        # Get count of points in collection
        count = 0
        offset = None
        batch_size = 100

        while True:
            result = vector_store.client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points = result[0]
            next_offset = result[1]

            count += len(points)

            if next_offset is None or len(points) == 0:
                break

            offset = next_offset

        return CollectionDetailsResponse(
            name=collection_name,
            count=count,
            metadata={},
            status="success",
        )

    except AppException:
        raise
    except Exception as e:
        logger.error(f"Error fetching collection details: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_INTERNAL_SERVER_ERROR",
            message=f"An unexpected error occurred: {str(e)}",
        ) from e
