"""
Embedding Service

Handles embedding generation for RAG.
Uses OpenAI embeddings to convert text chunks to vectors.
"""

from .service import EmbeddingService

__all__ = ["EmbeddingService"]
