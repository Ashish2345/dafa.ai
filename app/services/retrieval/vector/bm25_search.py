"""
BM25 Keyword Search Service

Implements BM25 algorithm for keyword-based search over document chunks.
BM25 is effective for exact term matching and complements vector search.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from loguru import logger
from rank_bm25 import BM25Okapi


class BM25SearchService:
    """
    BM25 keyword search service for document chunks.
    
    Uses BM25 algorithm to score chunks based on keyword relevance.
    Effective for exact term matching, section numbers, act names, etc.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 search service.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.indexes: Dict[str, BM25Okapi] = {}  # Collection name -> BM25 index
        self.chunk_corpus: Dict[str, List[Dict[str, Any]]] = {}  # Collection name -> chunks

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str,
    ) -> bool:
        """
        Build BM25 index for a collection of chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            collection_name: Collection name to index
            
        Returns:
            True if index built successfully
        """
        if not chunks:
            logger.warning(f"No chunks provided for BM25 index: {collection_name}")
            return False

        try:
            # Extract and tokenize chunk texts
            tokenized_corpus = []
            for chunk in chunks:
                text = chunk.get("text", "")
                tokens = self._tokenize(text)
                tokenized_corpus.append(tokens)

            # Build BM25 index
            bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
            self.indexes[collection_name] = bm25
            self.chunk_corpus[collection_name] = chunks

            logger.info(
                f"Built BM25 index for collection '{collection_name}' "
                f"with {len(chunks)} chunks"
            )
            return True

        except Exception as e:
            logger.error(f"Error building BM25 index for '{collection_name}': {e}")
            return False

    def search(
        self,
        query: str,
        collection_name: str,
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search chunks using BM25 keyword matching.
        
        Args:
            query: Search query string
            collection_name: Collection to search
            limit: Maximum number of results
            filter_conditions: Optional metadata filters
            
        Returns:
            List of chunks with BM25 scores, sorted by relevance
        """
        if collection_name not in self.indexes:
            logger.warning(f"BM25 index not found for collection: {collection_name}")
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            # Get BM25 scores
            bm25 = self.indexes[collection_name]
            scores = bm25.get_scores(query_tokens)

            # Get chunks
            chunks = self.chunk_corpus[collection_name]

            # Combine scores with chunks
            results = []
            for idx, (chunk, score) in enumerate(zip(chunks, scores)):
                # Apply metadata filters if provided
                if filter_conditions:
                    chunk_metadata = chunk.get("metadata", {})
                    if not self._matches_filters(chunk_metadata, filter_conditions):
                        continue

                results.append({
                    "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                    "text": chunk.get("text", ""),
                    "score": float(score),
                    "metadata": chunk.get("metadata", {}),
                    "collection": collection_name,
                    "search_type": "bm25",
                })

            # Sort by score (descending) and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:limit]

            top_score = results[0]['score'] if results else 0
            logger.debug(
                f"BM25 search in '{collection_name}': {len(results)} results "
                f"(top score: {top_score:.4f})"
            )
            return results

        except Exception as e:
            logger.error(f"Error in BM25 search for '{collection_name}': {e}")
            return []

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Extract words, numbers, and legal terms
        # Keep section numbers, act names, etc. as single tokens
        tokens = []

        # First, extract special patterns (section numbers, etc.)
        special_patterns = [
            r'section\s+\d+[a-z]?',  # section 12, section 12a
            r'chapter\s+\d+',
            r'part\s+\d+',
            r'article\s+\d+',
            r'clause\s+\([a-z0-9]+\)',  # clause (a)
            r'\d+[a-z]?',  # numbers with optional letter suffix
        ]

        # Extract and remove special patterns
        special_tokens = []
        for pattern in special_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            special_tokens.extend(matches)
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # Tokenize remaining text (words only)
        word_tokens = re.findall(r'\b[a-z]+\b', text)

        # Combine special tokens and word tokens
        tokens = special_tokens + word_tokens

        # Remove very short tokens (less than 2 chars) except numbers
        tokens = [
            t for t in tokens
            if len(t) >= 2 or re.match(r'^\d+', t)
        ]

        return tokens

    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filter_conditions: Dict[str, Any],
    ) -> bool:
        """
        Check if chunk metadata matches filter conditions.
        
        Args:
            metadata: Chunk metadata
            filter_conditions: Filter conditions to match
            
        Returns:
            True if metadata matches all filter conditions
        """
        for key, value in filter_conditions.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def remove_collection(self, collection_name: str) -> bool:
        """
        Remove BM25 index for a collection.
        
        Args:
            collection_name: Collection to remove
            
        Returns:
            True if removed successfully
        """
        if collection_name in self.indexes:
            del self.indexes[collection_name]
        if collection_name in self.chunk_corpus:
            del self.chunk_corpus[collection_name]
        logger.info(f"Removed BM25 index for collection: {collection_name}")
        return True

    def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a collection's BM25 index.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Dictionary with stats or None if collection not found
        """
        if collection_name not in self.indexes:
            return None

        chunks = self.chunk_corpus.get(collection_name, [])
        return {
            "collection_name": collection_name,
            "num_chunks": len(chunks),
            "indexed": True,
        }
