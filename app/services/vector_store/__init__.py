"""
Vector Store Service

Handles Qdrant vector database operations.
Stores and retrieves document chunks with embeddings.
"""

from .service import VectorStoreService

__all__ = ["VectorStoreService"]
