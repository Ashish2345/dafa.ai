"""
Vector retrieval strategy — traditional embeddings + Qdrant + BM25 hybrid.
"""

import re
from typing import Any

from loguru import logger

from app.services.retrieval.base import RetrievalStrategy, RetrievedChunk
from app.services.retrieval.vector.bm25_search import BM25SearchService
from app.services.retrieval.vector.embeddings import EmbeddingService
from app.services.retrieval.vector.hybrid_search import HybridSearchService
from app.services.retrieval.vector.reranker import Reranker
from app.services.retrieval.vector.vector_store import VectorStoreService


class VectorStrategy(RetrievalStrategy):
    """Traditional embeddings + vector search + BM25 hybrid + re-ranking."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStoreService | None = None,
        hybrid_search: HybridSearchService | None = None,
        reranker: Reranker | None = None,
        chunking_service: Any | None = None,
    ):
        self.embeddings = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreService()

        if hybrid_search:
            self.hybrid_search = hybrid_search
        else:
            from app.settings import settings
            bm25 = BM25SearchService()
            self.hybrid_search = HybridSearchService(
                vector_store=self.vector_store,
                bm25_service=bm25,
                vector_weight=getattr(settings, "hybrid_search_vector_weight", 0.7),
                bm25_weight=getattr(settings, "hybrid_search_bm25_weight", 0.3),
            )

        if reranker is not None:
            self.reranker = reranker
        else:
            from app.settings import settings
            self.reranker = Reranker(
                model_name=getattr(settings, "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                use_cross_encoder=True,
                top_k=getattr(settings, "rerank_top_k", 5),
            )

        self.chunking_service = chunking_service

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[RetrievedChunk]:
        query_embeddings = self.embeddings.generate_embeddings([query])
        if not query_embeddings or not query_embeddings[0]:
            logger.error("Failed to generate query embedding")
            return []
        query_embedding = query_embeddings[0]

        if collection_name:
            collection_names = [collection_name]
        else:
            collection_names = self.vector_store.get_all_collections()
            if not collection_names:
                return []

        from app.settings import settings
        retrieve_k = getattr(settings, "rerank_retrieve_k", 20)

        for cname in collection_names:
            if not self.hybrid_search.bm25_service.indexes.get(cname):
                self.hybrid_search.load_bm25_index_from_qdrant(cname)

        search_results = self.hybrid_search.search(
            query=query,
            query_embedding=query_embedding,
            collection_names=collection_names,
            limit=retrieve_k,
            filter_conditions=filter_conditions,
            retrieve_k=retrieve_k * 2,
        )

        if not search_results:
            return []

        if self.reranker:
            search_results = self.reranker.rerank(query=query, chunks=search_results, top_k=top_k)
        else:
            search_results = search_results[:top_k]

        return [
            RetrievedChunk(
                text=r.get("text", ""),
                source={
                    "document_name": r.get("metadata", {}).get("act_name", "Unknown"),
                    "section": ", ".join(r.get("metadata", {}).get("sections_in_chunk", [])),
                    "chunk_id": r.get("chunk_id", ""),
                },
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in search_results
        ]

    async def ingest(
        self,
        document_id: str,
        markdown: str,
        metadata: dict[str, Any],
    ) -> None:
        from app.services.ingestion.chunking import ChunkingService

        chunking = self.chunking_service or ChunkingService()
        chunks = chunking.chunk_document(markdown, metadata)

        chunk_texts = [c.get("text", "") for c in chunks]
        embeddings = self.embeddings.generate_embeddings(chunk_texts)
        if not embeddings:
            logger.error("Failed to generate embeddings during ingestion")
            return

        act_name = metadata.get("act_name", "")
        collection_name = re.sub(r"[^a-zA-Z0-9\s]", "", act_name).lower().strip()
        collection_name = re.sub(r"\s+", "_", collection_name) or "unknown_act"

        self.vector_store.store_chunks(
            chunks=chunks, embeddings=embeddings,
            document_id=document_id, collection_name=collection_name,
        )
        self.hybrid_search.build_bm25_index(chunks=chunks, collection_name=collection_name)
        logger.info(f"Vector ingestion complete: {len(chunks)} chunks in '{collection_name}'")
