"""
PageIndex repository — MongoDB storage for document trees and full Markdown content.

Collections:
  page_index_trees   — hierarchical tree JSON produced by PageIndexService
  page_index_content — full Markdown text keyed by document_id (for node text retrieval)
"""

from datetime import datetime, timezone
from typing import Optional

from loguru import logger


class PageIndexRepository:
    """CRUD for page_index_trees and page_index_content collections."""

    def __init__(self, database):
        self.trees = database.page_index_trees
        self.content = database.page_index_content

    async def save_tree(
        self,
        document_id: str,
        tree: dict,
        markdown: str,
        language: str = "en",
    ) -> None:
        """
        Upsert the hierarchical tree and full Markdown for a document.

        Args:
            document_id: Unique document identifier (matches documents collection)
            tree: Tree JSON returned by PageIndexService.build_tree()
            markdown: Full Markdown text of the document (used for node text retrieval)
            language: Document language ("en" | "ne")
        """
        now = datetime.now(timezone.utc)

        await self.trees.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "document_id": document_id,
                    "tree": tree,
                    "language": language,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

        await self.content.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "document_id": document_id,
                    "markdown": markdown,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

        logger.info(f"Saved PageIndex tree for document_id={document_id} (language={language})")

    async def get_tree(self, document_id: str) -> Optional[dict]:
        """Return the tree document for a given document_id, or None."""
        return await self.trees.find_one({"document_id": document_id}, {"_id": 0})

    async def get_markdown(self, document_id: str) -> Optional[str]:
        """Return the full Markdown string for a given document_id, or None."""
        doc = await self.content.find_one({"document_id": document_id}, {"_id": 0, "markdown": 1})
        return doc["markdown"] if doc else None

    async def list_document_ids(self) -> list[str]:
        """Return all document_ids that have a PageIndex tree."""
        cursor = self.trees.find({}, {"document_id": 1, "_id": 0})
        return [doc["document_id"] async for doc in cursor]
