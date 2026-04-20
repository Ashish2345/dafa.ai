"""
Document repository — consolidated document CRUD.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


class DocumentRepository:
    """CRUD for the documents collection."""

    def __init__(self, database):
        self.collection = database.documents

    async def save_initial(
        self,
        document_id: str,
        filename: str,
        category: str | None = None,
        title: str | None = None,
        user_id: str | None = None,
        scope: str = "public",
    ) -> None:
        """Create a stub record immediately so callers can poll status.

        Phase 14: ``user_id`` and ``scope`` make documents filterable as either
        a public catalog item (``scope='public'``, ``user_id=None``) or a
        private per-user upload (``scope='private'``, ``user_id=<owner>``).
        """
        now = datetime.now(timezone.utc)
        patch: dict[str, Any] = {
            "document_id": document_id,
            "filename": filename,
            "title": title or filename,
            "status": "processing",
            "progress_step": "Queued",
            "updated_at": now,
            "scope": scope,
            "user_id": user_id,
        }
        if category:
            patch["category"] = category
        await self.collection.update_one(
            {"document_id": document_id},
            {"$set": patch, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    async def update_status(
        self,
        document_id: str,
        status: str,
        step: str = "",
        error: str = "",
    ) -> None:
        """Update processing status and current step in-place."""
        patch: dict[str, Any] = {"status": status, "updated_at": datetime.now(timezone.utc)}
        if step:
            patch["progress_step"] = step
        if error:
            patch["error"] = error
        await self.collection.update_one({"document_id": document_id}, {"$set": patch})

    async def save(
        self,
        document_id: str,
        filename: str,
        metadata: dict[str, Any],
        strategy: str,
        custom_tree_provided: bool = False,
        ingest_warnings: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "document_id": document_id,
                    "filename": filename,
                    "metadata": metadata,
                    "strategy": strategy,
                    "status": "completed",
                    "progress_step": "Done",
                    "custom_tree_provided": custom_tree_provided,
                    "ingest_warnings": ingest_warnings,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def get(self, document_id: str) -> Optional[dict]:
        return await self.collection.find_one({"document_id": document_id}, {"_id": 0})

    async def list_all(
        self,
        skip: int = 0,
        limit: int = 50,
        category: str | None = None,
        user_id: str | None = None,
        scope: str | None = None,
    ) -> list[dict]:
        """List documents with visibility filtering (Phase 14).

        - Public catalog docs have ``scope='public'`` (or no scope field on
          legacy records) and are visible to everyone.
        - Private docs have ``scope='private'`` and ``user_id=<owner>`` — only
          the owner sees them.

        ``category='workspace'`` is a UX shorthand that the API layer now
        translates to ``scope='private', user_id=<me>`` — callers that pass
        both ``category`` and ``user_id``/``scope`` still get AND semantics.
        """
        query: dict[str, Any] = {}
        if category:
            query["category"] = category
        if scope == "private":
            query["scope"] = "private"
            if user_id is not None:
                query["user_id"] = user_id
        elif scope == "public":
            # Public + legacy (pre-Phase-14) docs with no scope field are both visible
            query["$or"] = [{"scope": "public"}, {"scope": {"$exists": False}}]
        elif user_id is not None:
            # Caller wants to see everything visible to them: public OR owned private
            query["$or"] = [
                {"scope": "public"},
                {"scope": {"$exists": False}},
                {"scope": "private", "user_id": user_id},
            ]

        cursor = self.collection.find(query, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit)
        return [doc async for doc in cursor]

    async def delete(self, document_id: str) -> bool:
        result = await self.collection.delete_one({"document_id": document_id})
        return result.deleted_count > 0
