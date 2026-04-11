"""
Vector Store Service

Manages Qdrant vector database operations for storing and retrieving document chunks.
"""

import hashlib
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from app.settings import settings


class VectorStoreService:
    """
    Service for managing Qdrant vector database operations.

    Handles collection creation, storing chunks with embeddings,
    and retrieving similar chunks.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        port: int = 6333,
        collection_name: str = "finance_acts",
        vector_size: int = 1536,  # OpenAI text-embedding-3-small dimension
        timeout: float = 300.0,  # 5 minutes default timeout
        batch_size: int = 100,  # Batch size for storing chunks
    ):
        """
        Initialize the vector store service.

        Args:
            url: Qdrant server URL (defaults to QDRANT_URL from settings or localhost)
            port: Qdrant server port (default: 6333)
            collection_name: Name of the Qdrant collection
            vector_size: Dimension of embedding vectors (default: 1536 for OpenAI)
            timeout: Request timeout in seconds (default: 300.0)
            batch_size: Number of chunks to store per batch (default: 100)
        """
        self.url = url or getattr(settings, "qdrant_url", "http://localhost")
        self.port = port or getattr(settings, "qdrant_port", 6333)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        self.batch_size = batch_size

        # Initialize Qdrant client with timeout
        try:
            self.client = QdrantClient(
                url=self.url,
                port=self.port,
                timeout=timeout,
            )
            logger.info(f"Connected to Qdrant at {self.url}:{self.port} (timeout: {timeout}s)")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None

    def ensure_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Ensure the collection exists, create if it doesn't.

        Args:
            collection_name: Collection name (uses self.collection_name if not provided)

        Returns:
            True if collection exists or was created successfully
        """
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False

        target_collection = collection_name or self.collection_name

        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if target_collection in collection_names:
                logger.debug(f"Collection '{target_collection}' already exists")
                return True

            # Create collection
            logger.info(f"Creating collection '{target_collection}' with vector size {self.vector_size}")
            self.client.create_collection(
                collection_name=target_collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Collection '{target_collection}' created successfully")
            return True

        except Exception as e:
            logger.error(f"Error ensuring collection '{target_collection}': {e}")
            return False

    def get_all_collections(self) -> List[str]:
        """
        Get list of all collection names.

        Returns:
            List of collection names
        """
        if not self.client:
            logger.error("Qdrant client not initialized")
            return []

        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return collection_names
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []

    def store_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """
        Store chunks with embeddings in Qdrant.

        Args:
            chunks: List of chunk dictionaries with text and metadata
            embeddings: List of embedding vectors (one per chunk)
            document_id: Document identifier
            collection_name: Collection name (uses self.collection_name if not provided)

        Returns:
            True if storage was successful
        """
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False

        if len(chunks) != len(embeddings):
            logger.error(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")
            return False

        target_collection = collection_name or self.collection_name

        # Ensure collection exists
        if not self.ensure_collection(collection_name=target_collection):
            return False

        try:
            # Prepare points for Qdrant
            points = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if not embedding:
                    logger.warning(f"Skipping chunk {idx} - no embedding")
                    continue

                # Generate a valid UUID for point ID
                # Use a hash of document_id + chunk_id to ensure consistency
                chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
                id_string = f"{document_id}_{chunk_id}"
                
                # Generate UUID from string hash (deterministic)
                # This ensures the same chunk always gets the same UUID
                hash_obj = hashlib.md5(id_string.encode())
                point_id = uuid.UUID(hash_obj.hexdigest())

                # Prepare payload (metadata)
                payload = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "text": chunk.get("text", ""),
                    **chunk.get("metadata", {}),
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            # Store chunks in batches to avoid timeout
            total_chunks = len(points)
            logger.info(f"Storing {total_chunks} chunks in Qdrant (batch size: {self.batch_size})")
            
            stored_count = 0
            failed_count = 0
            
            for batch_start in range(0, total_chunks, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_chunks)
                batch_points = points[batch_start:batch_end]
                batch_num = (batch_start // self.batch_size) + 1
                total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
                
                try:
                    logger.debug(f"Storing batch {batch_num}/{total_batches} ({len(batch_points)} chunks)")
                    self.client.upsert(
                        collection_name=target_collection,
                        points=batch_points,
                    )
                    stored_count += len(batch_points)
                    logger.debug(f"Successfully stored batch {batch_num}/{total_batches}")
                except Exception as batch_error:
                    failed_count += len(batch_points)
                    logger.error(f"Error storing batch {batch_num}/{total_batches}: {batch_error}")
                    # Continue with next batch instead of failing completely
                    continue

            if failed_count > 0:
                logger.warning(
                    f"Stored {stored_count}/{total_chunks} chunks for document {document_id}. "
                    f"Failed: {failed_count} chunks"
                )
                # Return True if at least some chunks were stored
                return stored_count > 0
            else:
                logger.info(f"Successfully stored all {stored_count} chunks for document {document_id}")
                return True

        except Exception as e:
            logger.error(f"Error storing chunks in Qdrant: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using a query embedding in a single collection.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            filter_conditions: Optional filter conditions (e.g., {"document_id": "doc123"})
            collection_name: Collection name (uses self.collection_name if not provided)

        Returns:
            List of search results with chunks and scores
        """
        if not self.client:
            logger.error("Qdrant client not initialized")
            return []

        target_collection = collection_name or self.collection_name

        try:
            # Build filter if conditions provided
            search_filter = None
            if filter_conditions:
                filter_conditions_list = []
                for key, value in filter_conditions.items():
                    filter_conditions_list.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value),
                        )
                    )
                if filter_conditions_list:
                    search_filter = models.Filter(must=filter_conditions_list)

            # Use query_points method (correct API for qdrant-client)
            # query_points accepts query as vector list directly
            results = self.client.query_points(
                collection_name=target_collection,
                query=query_embedding,  # Direct vector list
                limit=limit,
                query_filter=search_filter,
            )

            # Format results
            # query_points returns a QueryResponse object with .points attribute
            formatted_results = []
            result_points = results.points if hasattr(results, "points") else results
            
            for result in result_points:
                formatted_results.append({
                    "chunk_id": result.payload.get("chunk_id", ""),
                    "text": result.payload.get("text", ""),
                    "score": result.score if hasattr(result, "score") else 0.0,
                    "metadata": {k: v for k, v in result.payload.items() if k not in ["text", "chunk_id"]},
                    "collection": target_collection,  # Add collection name to results
                })

            logger.debug(f"Found {len(formatted_results)} similar chunks in '{target_collection}'")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching Qdrant collection '{target_collection}': {e}")
            return []

    def search_multiple_collections(
        self,
        query_embedding: List[float],
        collection_names: List[str],
        limit_per_collection: int = 5,
        total_limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks across multiple collections.

        Args:
            query_embedding: Query embedding vector
            collection_names: List of collection names to search
            limit_per_collection: Maximum results per collection
            total_limit: Maximum total results to return
            filter_conditions: Optional filter conditions

        Returns:
            List of search results with chunks and scores, sorted by score
        """
        if not self.client:
            logger.error("Qdrant client not initialized")
            return []

        if not collection_names:
            logger.warning("No collections specified for search")
            return []

        all_results = []

        for collection_name in collection_names:
            try:
                results = self.search(
                    query_embedding=query_embedding,
                    limit=limit_per_collection,
                    filter_conditions=filter_conditions,
                    collection_name=collection_name,
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Error searching collection '{collection_name}': {e}")
                continue

        # Sort by score (descending) and limit total results
        all_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        limited_results = all_results[:total_limit]

        logger.info(
            f"Searched {len(collection_names)} collections, "
            f"found {len(all_results)} total results, "
            f"returning top {len(limited_results)}"
        )

        return limited_results

    def get_chunks_for_collection(
        self,
        collection_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks from a collection for BM25 indexing.
        
        Args:
            collection_name: Collection name (uses self.collection_name if not provided)
            limit: Maximum number of chunks to retrieve (None for all)
            
        Returns:
            List of chunk dictionaries
        """
        if not self.client:
            logger.error("Qdrant client not initialized")
            return []

        target_collection = collection_name or self.collection_name

        try:
            # Scroll through all points in the collection
            chunks = []
            offset = None
            batch_size = 100

            while True:
                # Use scroll API to get points
                result = self.client.scroll(
                    collection_name=target_collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # Don't need vectors for BM25
                )

                points = result[0]  # Points
                next_offset = result[1]  # Next offset

                # Convert points to chunk format
                for point in points:
                    chunks.append({
                        "chunk_id": point.payload.get("chunk_id", ""),
                        "text": point.payload.get("text", ""),
                        "metadata": {k: v for k, v in point.payload.items() if k not in ["text", "chunk_id"]},
                    })

                # Check if we've retrieved enough or reached the end
                if limit and len(chunks) >= limit:
                    chunks = chunks[:limit]
                    break

                if next_offset is None:
                    break

                offset = next_offset

            logger.info(f"Retrieved {len(chunks)} chunks from collection '{target_collection}' for BM25 indexing")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks from collection '{target_collection}': {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document identifier

        Returns:
            True if deletion was successful
        """
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False

        try:
            # Delete points with matching document_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )

            logger.info(f"Deleted all chunks for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document from Qdrant: {e}")
            return False
