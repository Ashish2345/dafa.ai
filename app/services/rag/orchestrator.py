"""
RAG Query Orchestrator

Orchestrates the RAG query pipeline:
1. Generate query embedding
2. Search Qdrant for relevant chunks
3. Synthesize answer using LLM
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.rag.collection_router import CollectionRouter
from app.services.retrieval import (
    ContextOptimizer,
    HybridSearchService,
    LLMReranker,
    MetadataEnhancer,
    QueryEnhancer,
    Reranker,
)
from app.services.vector_store import VectorStoreService


class RAGOrchestrator:
    """
    Orchestrates RAG query pipeline.

    Handles the complete flow from user query to final answer.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreService] = None,
        llm_service: Optional[LLMService] = None,
        collection_router: Optional[CollectionRouter] = None,
        hybrid_search: Optional[HybridSearchService] = None,
        query_enhancer: Optional[QueryEnhancer] = None,
        metadata_enhancer: Optional[MetadataEnhancer] = None,
        context_optimizer: Optional[ContextOptimizer] = None,
        reranker: Optional[Reranker] = None,
        llm_reranker: Optional[LLMReranker] = None,
        top_k: int = 5,
        use_hybrid_search: bool = True,
        use_rerank: bool = True,
        use_metadata_enhancement: bool = True,
        use_context_optimization: bool = True,
    ):
        """
        Initialize the RAG orchestrator.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Service for vector search
            llm_service: Service for LLM calls
            collection_router: Service for routing queries to collections
            hybrid_search: Hybrid search service (vector + BM25)
            query_enhancer: Query enhancement service
            metadata_enhancer: Metadata enhancement service
            context_optimizer: Context optimization service
            reranker: Cross-encoder re-ranker service
            llm_reranker: LLM-based re-ranker service
            top_k: Number of top chunks to retrieve
            use_hybrid_search: Whether to use hybrid search (default: True)
            use_rerank: Whether to use re-ranking (default: True)
            use_metadata_enhancement: Whether to use metadata enhancement (default: True)
            use_context_optimization: Whether to use context optimization (default: True)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreService()
        self.llm_service = llm_service or LLMService()
        self.collection_router = collection_router or CollectionRouter(
            vector_store=self.vector_store,
            llm_service=self.llm_service,
        )
        self.query_enhancer = query_enhancer or QueryEnhancer()
        self.metadata_enhancer = metadata_enhancer or MetadataEnhancer()
        
        # Initialize context optimizer if enabled
        if use_context_optimization:
            from app.settings import settings
            max_context_tokens = getattr(settings, "max_context_tokens", 30000)
            reserved_tokens = getattr(settings, "context_reserved_tokens", 2000)
            self.context_optimizer = context_optimizer or ContextOptimizer(
                max_context_tokens=max_context_tokens,
                reserved_tokens=reserved_tokens,
            )
        else:
            self.context_optimizer = None
        
        self.use_hybrid_search = use_hybrid_search
        self.use_rerank = use_rerank
        self.use_metadata_enhancement = use_metadata_enhancement
        self.use_context_optimization = use_context_optimization
        
        # Initialize hybrid search if enabled
        if use_hybrid_search:
            from app.settings import settings
            from app.services.retrieval import BM25SearchService
            
            bm25_service = BM25SearchService()
            self.hybrid_search = hybrid_search or HybridSearchService(
                vector_store=self.vector_store,
                bm25_service=bm25_service,
                vector_weight=getattr(settings, "hybrid_search_vector_weight", 0.7),
                bm25_weight=getattr(settings, "hybrid_search_bm25_weight", 0.3),
            )
        else:
            self.hybrid_search = None
        
        # Initialize re-ranker if enabled
        if use_rerank:
            from app.settings import settings
            
            rerank_model = getattr(settings, "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            rerank_top_k = getattr(settings, "rerank_top_k", 5)
            
            self.reranker = reranker or Reranker(
                model_name=rerank_model,
                use_cross_encoder=True,
                top_k=rerank_top_k,
            )
            
            # Initialize LLM re-ranker if enabled
            if getattr(settings, "rerank_use_llm", False):
                logger.warning(
                    "LLM-based re-ranking is ENABLED. This will increase API costs. "
                    "Consider using cross-encoder re-ranking (default) for cost efficiency."
                )
                self.llm_reranker = llm_reranker or LLMReranker(
                    llm_service=self.llm_service,
                    top_k=rerank_top_k,
                )
            else:
                self.llm_reranker = None
        else:
            self.reranker = None
            self.llm_reranker = None
        
        self.top_k = top_k

    def query(
        self,
        user_query: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user query and return relevant answer.

        Args:
            user_query: User's question/query
            filter_conditions: Optional filters for vector search (e.g., {"act_name": "VAT Act"})
            use_llm: Whether to use LLM for answer synthesis (default: True)
            collection_name: Optional specific collection to search (overrides collection routing)

        Returns:
            Dictionary with:
            {
                "query": str,
                "chunks": List[Dict],  # Retrieved chunks
                "answer": str,  # LLM-generated answer (if use_llm=True)
                "sources": List[Dict],  # Source chunks with metadata
            }
        """
        logger.info(f"Processing query: {user_query[:100]}...")

        # Step 0: Enhance query (extract entities, expand terms)
        logger.debug("Step 0: Enhancing query")
        enhanced_query_info = self.query_enhancer.enhance_query(user_query)
        query_intent = self.query_enhancer.classify_intent(user_query)
        logger.debug(
            f"Query intent: {query_intent}, "
            f"entities: {enhanced_query_info.get('entities', {})}, "
            f"keywords: {len(enhanced_query_info.get('keywords', []))}"
        )

        # Step 1: Generate query embedding
        logger.debug("Step 1: Generating query embedding")
        try:
            query_embeddings = self.embedding_service.generate_embeddings([user_query])
            if not query_embeddings or not query_embeddings[0]:
                raise ValueError("Failed to generate query embedding")
            query_embedding = query_embeddings[0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return {
                "query": user_query,
                "chunks": [],
                "answer": "Error: Could not process query embedding.",
                "sources": [],
                "error": str(e),
            }

        # Step 2: Extract metadata from query (for filtering and boosting)
        query_metadata = {}
        if self.use_metadata_enhancement:
            logger.debug("Step 2: Extracting metadata from query")
            query_metadata = self.metadata_enhancer.extract_query_metadata(user_query)
            logger.debug(
                f"Extracted metadata: act={query_metadata.get('act_name')}, "
                f"sections={query_metadata.get('sections')}, "
                f"year={query_metadata.get('year')}"
            )

        # Step 2.5: Build metadata filters and merge with user-provided filters
        metadata_filters = None
        if self.use_metadata_enhancement and query_metadata:
            metadata_filters = self.metadata_enhancer.build_metadata_filter(query_metadata)
            if metadata_filters:
                # Merge with user-provided filters
                if filter_conditions:
                    filter_conditions = {**filter_conditions, **metadata_filters}
                else:
                    filter_conditions = metadata_filters
                logger.debug(f"Applied metadata filters: {metadata_filters}")

        # Step 3: Determine which collections to search
        logger.debug("Step 3: Routing query to collections")
        if collection_name:
            # Use specified collection if provided
            all_collections = self.vector_store.get_all_collections()
            if collection_name not in all_collections:
                return {
                    "query": user_query,
                    "chunks": [],
                    "answer": f"Error: Collection '{collection_name}' not found.",
                    "sources": [],
                    "error": "Collection not found",
                }
            collection_names = [collection_name]
            logger.info(f"Using specified collection: {collection_name}")
        else:
            try:
                collection_names = self.collection_router.route_query(user_query)
                logger.info(f"Query routed to {len(collection_names)} collection(s): {collection_names}")
            except Exception as e:
                logger.error(f"Error routing query to collections: {e}")
                # Fallback: get all collections
                collection_names = self.vector_store.get_all_collections()
                if not collection_names:
                    return {
                        "query": user_query,
                        "chunks": [],
                        "answer": "Error: No collections available in the knowledge base.",
                        "sources": [],
                        "error": "No collections found",
                    }

        # Step 4: Search Qdrant across relevant collections (hybrid or vector-only)
        # Determine how many chunks to retrieve (more if re-ranking is enabled)
        from app.settings import settings
        retrieve_k = getattr(settings, "rerank_retrieve_k", 20) if self.use_rerank else self.top_k
        
        logger.debug(
            f"Step 3: Searching {len(collection_names)} collection(s) "
            f"for top {retrieve_k if self.use_rerank else self.top_k} chunks"
        )
        try:
            if self.use_hybrid_search and self.hybrid_search:
                # Use hybrid search (vector + BM25)
                logger.debug("Using hybrid search (vector + BM25)")
                
                # Ensure BM25 indexes exist for collections
                for collection_name in collection_names:
                    if not self.hybrid_search.bm25_service.indexes.get(collection_name):
                        logger.debug(f"BM25 index not found for {collection_name}, loading from Qdrant...")
                        self.hybrid_search.load_bm25_index_from_qdrant(collection_name)
                
                search_results = self.hybrid_search.search(
                    query=user_query,
                    query_embedding=query_embedding,
                    collection_names=collection_names,
                    limit=retrieve_k if self.use_rerank else self.top_k,
                    filter_conditions=filter_conditions,
                    retrieve_k=retrieve_k * 2 if self.use_rerank else self.top_k * 2,  # Retrieve more for better combination
                )
            else:
                # Fallback to vector-only search
                logger.debug("Using vector-only search")
                search_limit = retrieve_k if self.use_rerank else self.top_k
                if len(collection_names) == 1:
                    # Single collection search
                    search_results = self.vector_store.search(
                        query_embedding=query_embedding,
                        limit=search_limit,
                        filter_conditions=filter_conditions,
                        collection_name=collection_names[0],
                    )
                else:
                    # Multi-collection search
                    limit_per_collection = max(2, search_limit // len(collection_names))
                    search_results = self.vector_store.search_multiple_collections(
                        query_embedding=query_embedding,
                        collection_names=collection_names,
                        limit_per_collection=limit_per_collection,
                        total_limit=search_limit,
                        filter_conditions=filter_conditions,
                    )

            if not search_results:
                logger.warning("No relevant chunks found")
                return {
                    "query": user_query,
                    "chunks": [],
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "searched_collections": collection_names,
                }

            logger.info(f"Found {len(search_results)} relevant chunks from {len(collection_names)} collection(s)")

            # Step 4.5: Apply metadata boosting (if enabled)
            if self.use_metadata_enhancement and search_results and query_metadata:
                logger.debug("Step 4.5: Applying metadata-based boosting")
                try:
                    boost_weight = getattr(settings, "metadata_boost_weight", 0.2)
                    search_results = self.metadata_enhancer.boost_by_metadata(
                        results=search_results,
                        query_metadata=query_metadata,
                        boost_weight=boost_weight,
                    )
                    logger.debug(f"Applied metadata boosting to {len(search_results)} results")
                except Exception as e:
                    logger.warning(f"Error in metadata boosting, continuing without boost: {e}")

            # Step 5: Re-rank retrieved chunks (if enabled)
            if self.use_rerank and search_results and self.reranker:
                logger.debug(f"Step 5: Re-ranking {len(search_results)} chunks to top {self.top_k}")
                try:
                    # Use LLM re-ranker for critical queries if enabled
                    if self.llm_reranker and query_intent in ["calculation", "complex", "legal_interpretation"]:
                        logger.info(
                            "Using LLM-based re-ranking for critical query (higher cost). "
                            f"Query intent: {query_intent}"
                        )
                        search_results = self.llm_reranker.rerank(
                            query=user_query,
                            chunks=search_results,
                            top_k=self.top_k,
                        )
                    else:
                        # Use cross-encoder re-ranking
                        search_results = self.reranker.rerank(
                            query=user_query,
                            chunks=search_results,
                            top_k=self.top_k,
                        )
                    logger.info(f"Re-ranked to {len(search_results)} top chunks")
                except Exception as e:
                    logger.warning(f"Error in re-ranking, using original results: {e}")
                    # Fallback: use original results (already limited to top_k)
                    search_results = search_results[:self.top_k]

        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return {
                "query": user_query,
                "chunks": [],
                "answer": "Error: Could not search the knowledge base.",
                "sources": [],
                "error": str(e),
                "searched_collections": collection_names,
            }

        # Step 6: Synthesize answer using LLM (if enabled)
        answer = ""
        answer_metadata: Dict[str, Any] = {}
        context_metadata: Dict[str, Any] = {}
        if use_llm:
            logger.debug("Step 6: Synthesizing answer with LLM")
            try:
                answer_result = self._synthesize_answer(user_query, search_results)
                # Handle tuple response (text, metadata) or string response
                if isinstance(answer_result, tuple):
                    if len(answer_result) == 2:
                        answer, answer_metadata = answer_result
                    elif len(answer_result) == 3:
                        answer, answer_metadata, context_metadata = answer_result
                    else:
                        answer = answer_result[0]
                        answer_metadata = {}
                else:
                    answer = answer_result
                    answer_metadata = {}
            except Exception as e:
                logger.error(f"Error synthesizing answer: {e}")
                answer = "Error: Could not generate answer."
                answer_metadata = {"is_error": True, "error": str(e)}

        # Prepare sources
        sources = [
            {
                "chunk_id": result["chunk_id"],
                "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "score": result["score"],
                "metadata": result["metadata"],
            }
            for result in search_results
        ]

        # Get metadata summary from results (if metadata enhancement is enabled)
        metadata_summary = {}
        if self.use_metadata_enhancement and search_results:
            try:
                metadata_summary = self.metadata_enhancer.get_metadata_summary(search_results)
            except Exception as e:
                logger.warning(f"Error generating metadata summary: {e}")

        # Build comprehensive result with metadata
        result = {
            "query": user_query,
            "chunks": search_results,
            "answer": answer if use_llm else "",
            "sources": sources,
            "searched_collections": collection_names,
            "metadata": {
                "chunks_retrieved": len(search_results),
                "collections_searched": len(collection_names),
                "collections": collection_names,
                "llm_used": use_llm,
                "query_metadata": query_metadata if self.use_metadata_enhancement else {},
                "metadata_summary": metadata_summary,
                "context_optimization": context_metadata if self.use_context_optimization else {},
            },
        }
        
        # Add answer metadata if LLM was used
        if use_llm and answer_metadata:
            result["metadata"]["answer_metadata"] = answer_metadata
        elif use_llm and answer:
            # Basic metadata for answer
            result["metadata"]["answer_metadata"] = {
                "length": len(answer),
                "word_count": len(answer.split()),
            }

        logger.info(
            f"Query processing complete: {len(search_results)} chunks, "
            f"{len(collection_names)} collections, "
            f"answer length: {len(result['answer'])} chars"
        )
        return result

    def _synthesize_answer(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> Union[str, Tuple[str, Dict[str, Any]], Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """
        Synthesize answer from retrieved chunks using LLM.

        Args:
            query: User query
            chunks: Retrieved chunks with text and metadata

        Returns:
            Generated answer (or tuple with metadata if context optimization is enabled)
        """
        context_metadata = {}
        
        # Optimize context if enabled
        if self.use_context_optimization and self.context_optimizer:
            logger.debug("Optimizing context window for LLM call")
            try:
                context, context_metadata = self.context_optimizer.build_optimized_context(
                    chunks=chunks,
                    query=query,
                )
                logger.debug(
                    f"Context optimization: {context_metadata['selected']}/{context_metadata['total']} chunks, "
                    f"{context_metadata['tokens_used']}/{context_metadata['tokens_available']} tokens "
                    f"({context_metadata['utilization']:.1%} utilization)"
                )
            except Exception as e:
                logger.warning(f"Error in context optimization, using all chunks: {e}")
                # Fallback: build context from all chunks
                context_parts = []
                for idx, chunk in enumerate(chunks, 1):
                    chunk_text = chunk.get("text", "")
                    metadata = chunk.get("metadata", {})
                    act_name = metadata.get("act_name", "Unknown Act")
                    section = metadata.get("sections_in_chunk", [])
                    section_str = ", ".join(section) if section else "N/A"

                    context_parts.append(
                        f"[Source {idx}]\n"
                        f"Act: {act_name}\n"
                        f"Section(s): {section_str}\n"
                        f"Content: {chunk_text}\n"
                    )
                context = "\n\n".join(context_parts)
                context_metadata = {"error": str(e), "fallback": True}
        else:
            # Build context from all chunks (no optimization)
            context_parts = []
            for idx, chunk in enumerate(chunks, 1):
                chunk_text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                act_name = metadata.get("act_name", "Unknown Act")
                section = metadata.get("sections_in_chunk", [])
                section_str = ", ".join(section) if section else "N/A"

                context_parts.append(
                    f"[Source {idx}]\n"
                    f"Act: {act_name}\n"
                    f"Section(s): {section_str}\n"
                    f"Content: {chunk_text}\n"
                )

            context = "\n\n".join(context_parts)

        # Build prompt
        system_instruction = """You are a helpful assistant that answers questions about finance acts and regulations.
Use the provided context to answer the user's question accurately and concisely.
If the context doesn't contain enough information to answer the question, say so.
Always cite the relevant Act and Section numbers when possible."""

        user_prompt = f"""Context from finance acts:

{context}

Question: {query}

Please provide a clear and accurate answer based on the context above. Include relevant Act names and Section numbers when available."""
        # Call LLM with metadata
        # Use higher max_tokens for detailed responses (tax calculations, etc.)
        # Gemini 2.5 Flash supports up to 8192 output tokens
        answer_result = self.llm_service.call(
            prompt=user_prompt,
            system_instruction=system_instruction,
            temperature=0.3,  # Lower temperature for factual answers
            max_tokens=8192,  # Increased from 1024 to allow complete detailed responses
            return_metadata=True,  # Get metadata for better response handling
        )

        # Handle tuple response (text, metadata) or string response
        if isinstance(answer_result, tuple):
            answer, llm_metadata = answer_result
            # Return answer with both LLM metadata and context metadata
            if context_metadata:
                return answer, llm_metadata, context_metadata
            return answer, llm_metadata
        else:
            # Return answer with context metadata if available
            if context_metadata:
                return answer_result, {}, context_metadata
            return answer_result
