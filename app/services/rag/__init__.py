"""
RAG Service

Orchestrates retrieval-augmented generation for finance act queries.
"""

from .collection_router import CollectionRouter
from .orchestrator import RAGOrchestrator

__all__ = ["RAGOrchestrator", "CollectionRouter"]
