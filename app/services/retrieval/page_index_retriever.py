"""
PageIndex Retriever — orchestrates vectorless RAG retrieval.

Given a user query:
  1. Loads all document trees from MongoDB (or filters by act name)
  2. For each document, runs LLM tree navigation to find relevant nodes
  3. Returns a flat list of section dicts (nodeId, title, text, metadata)
"""

import asyncio
from typing import Any, Optional

from loguru import logger

from app.db.mongodb import get_database
from app.db.repositories.page_index_repository import PageIndexRepository
from app.services.page_index.service import PageIndexService


class PageIndexRetriever:
    """
    Retrieves relevant document sections using the PageIndex vectorless approach.

    Compatible with the RAGOrchestrator's chunk interface:
    each returned item is a dict with "text", "metadata", and "chunk_id" keys.
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self.page_index_service = PageIndexService()

    async def retrieve(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: int = 5,
        db=None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant sections for a query using PageIndex tree navigation.

        Args:
            query: User question
            collection_name: Optional act name filter (e.g. "income_tax_act").
                             If None, searches all ingested documents.
            top_k: Maximum sections to return per document
            db: MongoDB database instance (injected or fetched)

        Returns:
            List of chunk-like dicts compatible with RAGOrchestrator._synthesize_answer():
            [{
                "chunk_id": str,
                "text": str,
                "nodeId": str,
                "title": str,
                "metadata": {"act_name": str, "node_id": str, "title": str, "source": "page_index"},
            }]
        """
        if db is None:
            db = await get_database()

        repo = PageIndexRepository(db)
        document_ids = await repo.list_document_ids()

        if not document_ids:
            logger.warning("No PageIndex trees found in database")
            return []

        # Filter by act/collection name if provided
        if collection_name:
            # Heuristic: match document_id or act_name substring
            # In the future this could be a proper index lookup
            filtered = await self._filter_by_collection(document_ids, collection_name, repo)
            if filtered:
                document_ids = filtered

        all_sections: list[dict[str, Any]] = []

        for doc_id in document_ids:
            tree_doc = await repo.get_tree(doc_id)
            markdown = await repo.get_markdown(doc_id)

            if not tree_doc or not markdown:
                continue

            tree = tree_doc.get("tree", {})
            language = tree_doc.get("language", self.language)
            act_name = tree.get("document_title", doc_id)

            sections = self.page_index_service.retrieve_sections(
                query=query,
                tree=tree,
                markdown_content=markdown,
                language=language,
                top_k=top_k,
            )

            # Normalize to chunk-like format expected by RAGOrchestrator
            for section in sections:
                all_sections.append(
                    {
                        "chunk_id": f"{doc_id}::{section['nodeId']}",
                        "text": section["text"],
                        "nodeId": section["nodeId"],
                        "title": section["title"],
                        "metadata": {
                            **section.get("metadata", {}),
                            "act_name": act_name,
                            "document_id": doc_id,
                            "sections_in_chunk": [section["nodeId"]],
                        },
                    }
                )

        logger.info(f"PageIndex retrieval returned {len(all_sections)} sections for query: {query[:80]!r}")
        return all_sections[:top_k]

    async def _filter_by_collection(
        self, document_ids: list[str], collection_name: str, repo: PageIndexRepository
    ) -> list[str]:
        """Return document_ids whose act_name matches the collection_name hint."""
        matched = []
        for doc_id in document_ids:
            tree_doc = await repo.get_tree(doc_id)
            if not tree_doc:
                continue
            title = tree_doc.get("tree", {}).get("document_title", "")
            # Normalize both for comparison
            if collection_name.lower().replace("_", " ") in title.lower():
                matched.append(doc_id)
        return matched
