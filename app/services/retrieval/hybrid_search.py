"""
Hybrid Search Service

Combines vector similarity search with BM25 keyword search for improved retrieval.
Uses weighted combination of scores from both methods.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.retrieval.bm25_search import BM25SearchService
from app.services.vector_store import VectorStoreService


class HybridSearchService:
    """
    Hybrid search service combining vector and BM25 search.
    
    Combines semantic similarity (vector) with keyword matching (BM25)
    for more accurate retrieval, especially for finance act documents.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        bm25_service: Optional[BM25SearchService] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        """
        Initialize hybrid search service.
        
        Args:
            vector_store: Vector store service for semantic search
            bm25_service: BM25 service for keyword search
            vector_weight: Weight for vector search scores (default: 0.7)
            bm25_weight: Weight for BM25 search scores (default: 0.3)
        """
        self.vector_store = vector_store or VectorStoreService()
        self.bm25_service = bm25_service or BM25SearchService()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # Normalize weights
        total_weight = vector_weight + bm25_weight
        if total_weight > 0:
            self.vector_weight = vector_weight / total_weight
            self.bm25_weight = bm25_weight / total_weight
        else:
            self.vector_weight = 0.7
            self.bm25_weight = 0.3

    def search(
        self,
        query: str,
        query_embedding: List[float],
        collection_names: List[str],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        retrieve_k: int = 20,  # Retrieve more, then re-rank
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and BM25 results.
        
        Args:
            query: User query string
            query_embedding: Query embedding vector
            collection_names: List of collection names to search
            limit: Maximum number of results to return
            filter_conditions: Optional metadata filters
            retrieve_k: Number of results to retrieve from each method before combining
            
        Returns:
            List of chunks with hybrid scores, sorted by relevance
        """
        if not collection_names:
            logger.warning("No collections specified for hybrid search")
            return []

        all_results = {}

        # Perform vector search
        try:
            vector_results = self._vector_search(
                query_embedding=query_embedding,
                collection_names=collection_names,
                limit=retrieve_k,
                filter_conditions=filter_conditions,
            )
            logger.debug(f"Vector search returned {len(vector_results)} results")
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            vector_results = []

        # Perform BM25 search
        try:
            bm25_results = self._bm25_search(
                query=query,
                collection_names=collection_names,
                limit=retrieve_k,
                filter_conditions=filter_conditions,
            )
            logger.debug(f"BM25 search returned {len(bm25_results)} results")
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            bm25_results = []

        # Combine results
        combined_results = self._combine_results(
            vector_results=vector_results,
            bm25_results=bm25_results,
            limit=limit,
        )

        logger.info(
            f"Hybrid search: {len(vector_results)} vector + {len(bm25_results)} BM25 "
            f"= {len(combined_results)} combined results"
        )

        return combined_results

    def _vector_search(
        self,
        query_embedding: List[float],
        collection_names: List[str],
        limit: int,
        filter_conditions: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Perform vector search across collections."""
        if len(collection_names) == 1:
            results = self.vector_store.search(
                query_embedding=query_embedding,
                limit=limit,
                filter_conditions=filter_conditions,
                collection_name=collection_names[0],
            )
        else:
            limit_per_collection = max(2, limit // len(collection_names))
            results = self.vector_store.search_multiple_collections(
                query_embedding=query_embedding,
                collection_names=collection_names,
                limit_per_collection=limit_per_collection,
                total_limit=limit,
                filter_conditions=filter_conditions,
            )

        # Normalize vector scores (typically 0-1 for cosine similarity)
        # Qdrant returns cosine similarity scores (0-1 range)
        for result in results:
            result["vector_score"] = result.get("score", 0.0)
            result["search_type"] = "vector"

        return results

    def _bm25_search(
        self,
        query: str,
        collection_names: List[str],
        limit: int,
        filter_conditions: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Perform BM25 search across collections."""
        all_results = []

        for collection_name in collection_names:
            try:
                results = self.bm25_service.search(
                    query=query,
                    collection_name=collection_name,
                    limit=limit,
                    filter_conditions=filter_conditions,
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"BM25 search failed for '{collection_name}': {e}")
                continue

        # Normalize BM25 scores (BM25 scores can vary widely)
        if all_results:
            max_score = max(r["score"] for r in all_results)
            min_score = min(r["score"] for r in all_results)
            score_range = max_score - min_score if max_score > min_score else 1.0

            for result in all_results:
                # Normalize to 0-1 range
                normalized_score = (
                    (result["score"] - min_score) / score_range
                    if score_range > 0
                    else 0.0
                )
                result["bm25_score"] = normalized_score
                result["vector_score"] = 0.0  # No vector score for BM25-only results

        return all_results

    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and BM25 results with weighted scoring.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            limit: Maximum results to return
            
        Returns:
            Combined and re-ranked results
        """
        # Create a map of chunk_id -> result for deduplication
        combined_map = {}

        # Add vector results
        for result in vector_results:
            chunk_id = result.get("chunk_id", "")
            if chunk_id:
                combined_map[chunk_id] = result.copy()
                combined_map[chunk_id]["vector_score"] = result.get("vector_score", result.get("score", 0.0))
                combined_map[chunk_id]["bm25_score"] = 0.0  # Will be updated if found in BM25

        # Add/update with BM25 results
        for result in bm25_results:
            chunk_id = result.get("chunk_id", "")
            if chunk_id in combined_map:
                # Update existing result with BM25 score
                combined_map[chunk_id]["bm25_score"] = result.get("bm25_score", result.get("score", 0.0))
            else:
                # New result from BM25 only
                combined_map[chunk_id] = result.copy()
                combined_map[chunk_id]["vector_score"] = 0.0
                combined_map[chunk_id]["bm25_score"] = result.get("bm25_score", result.get("score", 0.0))

        # Calculate hybrid scores
        combined_results = []
        for chunk_id, result in combined_map.items():
            # Weighted combination
            hybrid_score = (
                self.vector_weight * result.get("vector_score", 0.0) +
                self.bm25_weight * result.get("bm25_score", 0.0)
            )

            result["score"] = hybrid_score
            result["search_type"] = "hybrid"
            combined_results.append(result)

        # Sort by hybrid score (descending)
        combined_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Return top results
        return combined_results[:limit]

    def build_bm25_index(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str,
    ) -> bool:
        """
        Build BM25 index for a collection.
        
        Args:
            chunks: List of chunks to index
            collection_name: Collection name
            
        Returns:
            True if index built successfully
        """
        return self.bm25_service.build_index(chunks, collection_name)

    def load_bm25_index_from_qdrant(
        self,
        collection_name: str,
        limit: Optional[int] = None,
    ) -> bool:
        """
        Load chunks from Qdrant and build BM25 index.
        
        Useful for rebuilding index or loading existing collections.
        
        Args:
            collection_name: Collection name
            limit: Maximum chunks to load (None for all)
            
        Returns:
            True if index built successfully
        """
        try:
            # Get chunks from Qdrant
            chunks = self.vector_store.get_chunks_for_collection(
                collection_name=collection_name,
                limit=limit,
            )

            if not chunks:
                logger.warning(f"No chunks found in Qdrant for collection: {collection_name}")
                return False

            # Build BM25 index
            return self.bm25_service.build_index(chunks, collection_name)

        except Exception as e:
            logger.error(f"Error loading BM25 index from Qdrant: {e}")
            return False
