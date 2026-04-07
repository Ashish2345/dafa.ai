"""
Re-ranking Service

Re-ranks retrieved chunks using cross-encoder models for better relevance.
Cross-encoders provide more accurate relevance scores than bi-encoders.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from app.prompts.old_flow import reranking as reranking_prompts


class Reranker:
    """
    Re-ranks retrieved chunks using cross-encoder models.
    
    Cross-encoders encode query and chunk together, providing
    more accurate relevance scores than bi-encoder embeddings.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cross_encoder: bool = True,
        top_k: int = 5,
    ):
        """
        Initialize the re-ranker.
        
        Args:
            model_name: Cross-encoder model name (default: ms-marco-MiniLM-L-6-v2)
            use_cross_encoder: Whether to use cross-encoder (default: True)
            top_k: Number of top results to return after re-ranking
        """
        self.model_name = model_name
        self.use_cross_encoder = use_cross_encoder
        self.top_k = top_k
        self.model = None

        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name)
                logger.info(f"Loaded cross-encoder model: {model_name}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not available. "
                    "Re-ranking will use fallback method."
                )
                self.use_cross_encoder = False
            except Exception as e:
                logger.error(f"Error loading cross-encoder model: {e}")
                self.use_cross_encoder = False

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks based on query relevance.
        
        Args:
            query: User query
            chunks: List of chunks to re-rank
            top_k: Number of top results to return (uses self.top_k if None)
            
        Returns:
            Re-ranked list of chunks
        """
        if not chunks:
            return []

        top_k = top_k or self.top_k

        # If no model available, return original chunks
        if not self.use_cross_encoder or not self.model:
            logger.debug("Cross-encoder not available, returning original ranking")
            return chunks[:top_k]

        try:
            # Prepare query-chunk pairs for cross-encoder
            pairs = []
            for chunk in chunks:
                chunk_text = chunk.get("text", "")
                # Truncate very long chunks to avoid token limits
                max_chunk_length = 512  # Conservative limit for cross-encoder
                if len(chunk_text) > max_chunk_length:
                    chunk_text = chunk_text[:max_chunk_length] + "..."
                pairs.append([query, chunk_text])

            # Get relevance scores from cross-encoder
            scores = self.model.predict(pairs)

            # Combine scores with chunks
            reranked_chunks = []
            for chunk, score in zip(chunks, scores):
                reranked_chunk = chunk.copy()
                reranked_chunk["rerank_score"] = float(score)
                # Update main score with re-ranking score
                # Combine original score with rerank score (weighted)
                original_score = chunk.get("score", 0.0)
                combined_score = 0.7 * float(score) + 0.3 * original_score
                reranked_chunk["score"] = combined_score
                reranked_chunks.append(reranked_chunk)

            # Sort by rerank score (descending)
            reranked_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

            # Return top-k
            result = reranked_chunks[:top_k]

            top_score = result[0].get('rerank_score', 0) if result else 0
            logger.debug(
                f"Re-ranked {len(chunks)} chunks to top {len(result)} "
                f"(top rerank score: {top_score:.4f})"
            )

            return result

        except Exception as e:
            logger.error(f"Error in re-ranking: {e}")
            # Fallback: return original chunks
            return chunks[:top_k]


class LLMReranker:
    """
    LLM-based re-ranking for critical queries.
    
    Uses LLM to score chunk relevance. More expensive but potentially
    more accurate for complex queries.
    
    WARNING: This is DISABLED by default due to higher API costs.
    Only enable if you need maximum accuracy for complex queries.
    """

    def __init__(
        self,
        llm_service: Optional[Any] = None,
        top_k: int = 5,
    ):
        """
        Initialize LLM-based re-ranker.
        
        Args:
            llm_service: LLM service for scoring
            top_k: Number of top results to return
            
        Note:
            This re-ranker makes additional LLM API calls, increasing costs.
            Consider using cross-encoder re-ranking (default) for cost efficiency.
        """
        self.llm_service = llm_service
        self.top_k = top_k
        if llm_service:
            logger.warning(
                "LLMReranker initialized. This will make additional LLM API calls "
                "for each query, increasing costs. Use only when necessary."
            )

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using LLM-based scoring.
        
        Args:
            query: User query
            chunks: List of chunks to re-rank
            top_k: Number of top results to return
            
        Returns:
            Re-ranked list of chunks
        """
        if not chunks or not self.llm_service:
            return chunks[:top_k or self.top_k]

        top_k = top_k or self.top_k

        try:
            # Build prompt for LLM to score chunks
            chunk_texts = []
            for idx, chunk in enumerate(chunks):
                chunk_text = chunk.get("text", "")[:500]  # Limit length
                chunk_texts.append(f"[Chunk {idx + 1}]\n{chunk_text}")

            chunks_text = "\n\n".join(chunk_texts)

            system_instruction, user_prompt_template = reranking_prompts.get_prompts()
            user_prompt = user_prompt_template.format(query=query, chunks_text=chunks_text)

            # Call LLM
            response = self.llm_service.call(
                prompt=user_prompt,
                system_instruction=system_instruction,
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=200,
            )

            # Parse scores from response
            import json
            try:
                # Extract JSON array from response
                response = response.strip()
                # Remove markdown code blocks if present
                if response.startswith("```"):
                    response = response.split("```")[1]
                    if response.startswith("json"):
                        response = response[4:]
                response = response.strip()

                scores = json.loads(response)
                if not isinstance(scores, list) or len(scores) != len(chunks):
                    raise ValueError("Invalid scores format")
            except Exception as e:
                logger.warning(f"Error parsing LLM scores: {e}. Using fallback scoring.")
                # Fallback: use original scores
                return chunks[:top_k]

            # Combine scores with chunks
            reranked_chunks = []
            for chunk, score in zip(chunks, scores):
                reranked_chunk = chunk.copy()
                reranked_chunk["rerank_score"] = float(score)
                # Update main score
                original_score = chunk.get("score", 0.0)
                combined_score = 0.7 * float(score) + 0.3 * original_score
                reranked_chunk["score"] = combined_score
                reranked_chunks.append(reranked_chunk)

            # Sort by rerank score
            reranked_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

            return reranked_chunks[:top_k]

        except Exception as e:
            logger.error(f"Error in LLM re-ranking: {e}")
            return chunks[:top_k]
