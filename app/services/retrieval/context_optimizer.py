"""
Context Window Optimizer

Optimizes context for LLM calls by:
- Token counting and estimation
- Smart chunk selection within token limits
- Context compression for long contexts
- Prioritizing high-quality chunks
"""

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class ContextOptimizer:
    """
    Optimizes context window for LLM calls.
    
    Ensures context fits within token limits while maximizing
    relevance and information density.
    """

    def __init__(
        self,
        max_context_tokens: int = 30000,  # Gemini 2.5 Flash context window
        reserved_tokens: int = 2000,  # For prompt, system instruction, query
        token_estimate_ratio: float = 0.25,  # ~4 chars per token estimate
    ):
        """
        Initialize the context optimizer.
        
        Args:
            max_context_tokens: Maximum tokens for context (default: 30000 for Gemini 2.5 Flash)
            reserved_tokens: Tokens reserved for prompt/system/query (default: 2000)
            token_estimate_ratio: Ratio for estimating tokens from characters (default: 0.25 = 4 chars/token)
        """
        self.max_context_tokens = max_context_tokens
        self.reserved_tokens = reserved_tokens
        self.token_estimate_ratio = token_estimate_ratio
        self.available_tokens = max_context_tokens - reserved_tokens

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        
        Uses a simple character-based estimation (more accurate than word count).
        Most tokenizers use ~4 characters per token on average.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        # Rough estimate: ~4 characters per token
        return int(len(text) * self.token_estimate_ratio)

    def select_optimal_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        max_tokens: Optional[int] = None,
        prioritize_quality: bool = True,
        prioritize_relevance: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Select optimal chunks that fit within token limits.
        
        Uses a greedy algorithm to select chunks based on:
        - Relevance score
        - Quality score
        - Token efficiency (information density)
        
        Args:
            chunks: List of chunks to select from
            query: User query (for context)
            max_tokens: Maximum tokens for context (uses self.available_tokens if None)
            prioritize_quality: Whether to prioritize high-quality chunks
            prioritize_relevance: Whether to prioritize high-relevance chunks
            
        Returns:
            Tuple of (selected_chunks, selection_metadata)
        """
        if not chunks:
            return [], {"selected": 0, "total": 0, "tokens_used": 0, "tokens_available": 0}

        max_tokens = max_tokens or self.available_tokens

        # Calculate token costs and scores for each chunk
        chunk_data = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            
            # Format chunk for context (same format as in _synthesize_answer)
            act_name = metadata.get("act_name", "Unknown Act")
            section = metadata.get("sections_in_chunk", [])
            section_str = ", ".join(section) if section else "N/A"
            
            formatted_chunk = (
                f"[Source]\n"
                f"Act: {act_name}\n"
                f"Section(s): {section_str}\n"
                f"Content: {chunk_text}\n"
            )
            
            tokens = self.estimate_tokens(formatted_chunk)
            
            # Calculate composite score
            score = chunk.get("score", 0.0)
            quality_score = chunk.get("quality_score", 0.5)  # Default to 0.5 if not present
            rerank_score = chunk.get("rerank_score", score)  # Use rerank score if available
            
            # Composite score: weighted combination
            if prioritize_quality and prioritize_relevance:
                composite_score = 0.5 * rerank_score + 0.3 * score + 0.2 * quality_score
            elif prioritize_quality:
                composite_score = 0.6 * quality_score + 0.4 * rerank_score
            elif prioritize_relevance:
                composite_score = 0.7 * rerank_score + 0.3 * score
            else:
                composite_score = rerank_score
            
            # Information density: score per token
            density = composite_score / max(tokens, 1)
            
            chunk_data.append({
                "chunk": chunk,
                "tokens": tokens,
                "score": composite_score,
                "density": density,
                "formatted": formatted_chunk,
            })

        # Sort by composite score (descending)
        chunk_data.sort(key=lambda x: x["score"], reverse=True)

        # Greedy selection: add chunks until token limit
        selected = []
        tokens_used = 0
        separator_tokens = 2  # "\n\n" between chunks

        for data in chunk_data:
            chunk_tokens = data["tokens"]
            
            # Check if adding this chunk would exceed limit
            if tokens_used + chunk_tokens + separator_tokens > max_tokens:
                # Try to fit if it's the first chunk (must include at least one)
                if not selected:
                    # Truncate first chunk if necessary
                    if chunk_tokens > max_tokens:
                        logger.warning(
                            f"First chunk ({chunk_tokens} tokens) exceeds limit ({max_tokens} tokens). "
                            "Truncating..."
                        )
                        # Truncate formatted chunk
                        max_chars = int((max_tokens - separator_tokens) / self.token_estimate_ratio)
                        data["formatted"] = data["formatted"][:max_chars] + "..."
                        chunk_tokens = max_tokens - separator_tokens
                    selected.append(data)
                    tokens_used += chunk_tokens + separator_tokens
                break
            
            selected.append(data)
            tokens_used += chunk_tokens + separator_tokens

        # Extract selected chunks
        selected_chunks = [data["chunk"] for data in selected]

        metadata = {
            "selected": len(selected_chunks),
            "total": len(chunks),
            "tokens_used": tokens_used,
            "tokens_available": max_tokens,
            "utilization": tokens_used / max_tokens if max_tokens > 0 else 0.0,
            "avg_score": sum(d["score"] for d in selected) / len(selected) if selected else 0.0,
        }

        logger.debug(
            f"Selected {len(selected_chunks)}/{len(chunks)} chunks "
            f"({tokens_used}/{max_tokens} tokens, {metadata['utilization']:.1%} utilization)"
        )

        return selected_chunks, metadata

    def build_optimized_context(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build optimized context string from chunks.
        
        Args:
            chunks: List of chunks
            query: User query
            max_tokens: Maximum tokens for context
            
        Returns:
            Tuple of (context_string, metadata)
        """
        # Select optimal chunks
        selected_chunks, selection_metadata = self.select_optimal_chunks(
            chunks=chunks,
            query=query,
            max_tokens=max_tokens,
        )

        # Build context string
        context_parts = []
        for idx, chunk in enumerate(selected_chunks, 1):
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

        # Estimate final token count
        context_tokens = self.estimate_tokens(context)
        selection_metadata["context_tokens"] = context_tokens

        return context, selection_metadata

    def compress_context(
        self,
        context: str,
        target_tokens: int,
        preserve_structure: bool = True,
    ) -> str:
        """
        Compress context to fit within target token limit.
        
        Uses simple truncation with structure preservation.
        For more advanced compression, could use summarization.
        
        Args:
            context: Context string to compress
            target_tokens: Target token count
            preserve_structure: Whether to preserve source structure
            
        Returns:
            Compressed context string
        """
        current_tokens = self.estimate_tokens(context)
        
        if current_tokens <= target_tokens:
            return context

        # Calculate compression ratio
        ratio = target_tokens / current_tokens
        
        # Simple truncation (could be enhanced with summarization)
        target_chars = int(len(context) * ratio)
        
        if preserve_structure:
            # Try to truncate at source boundaries
            sources = context.split("\n\n[Source")
            compressed_sources = []
            chars_used = 0
            
            for source in sources:
                if not source.strip():
                    continue
                    
                source_text = f"[Source{source}" if not source.startswith("[Source") else source
                source_tokens = self.estimate_tokens(source_text)
                
                if chars_used + len(source_text) <= target_chars:
                    compressed_sources.append(source_text)
                    chars_used += len(source_text)
                else:
                    # Truncate last source if needed
                    remaining = target_chars - chars_used
                    if remaining > 100:  # Only if meaningful space remains
                        truncated = source_text[:remaining] + "..."
                        compressed_sources.append(truncated)
                    break
            
            compressed = "\n\n".join(compressed_sources)
        else:
            # Simple truncation
            compressed = context[:target_chars] + "..."

        logger.debug(
            f"Compressed context: {current_tokens} -> {self.estimate_tokens(compressed)} tokens "
            f"(target: {target_tokens})"
        )

        return compressed

    def validate_context_size(
        self,
        context: str,
        query: str,
        system_instruction: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that context fits within token limits.
        
        Args:
            context: Context string
            query: User query
            system_instruction: Optional system instruction
            
        Returns:
            Tuple of (is_valid, validation_metadata)
        """
        context_tokens = self.estimate_tokens(context)
        query_tokens = self.estimate_tokens(query)
        system_tokens = self.estimate_tokens(system_instruction or "")
        
        total_tokens = context_tokens + query_tokens + system_tokens + self.reserved_tokens
        
        is_valid = total_tokens <= self.max_context_tokens
        
        metadata = {
            "is_valid": is_valid,
            "context_tokens": context_tokens,
            "query_tokens": query_tokens,
            "system_tokens": system_tokens,
            "total_tokens": total_tokens,
            "max_tokens": self.max_context_tokens,
            "utilization": total_tokens / self.max_context_tokens if self.max_context_tokens > 0 else 0.0,
        }

        if not is_valid:
            logger.warning(
                f"Context exceeds token limit: {total_tokens}/{self.max_context_tokens} tokens "
                f"({metadata['utilization']:.1%} utilization)"
            )

        return is_valid, metadata
