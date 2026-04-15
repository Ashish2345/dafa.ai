"""
Starred responses repository — MongoDB CRUD for per-user pinned Q&A answers.

Stores one document per starred response in `starred_responses`, scoped by `user_id`.
Maximum 50 starred items per user (oldest evicted on new adds).
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


class StarredRepository:
    """CRUD for the `starred_responses` MongoDB collection."""

    def __init__(self, database):
        self.collection = database.starred_responses

    async def create(
        self,
        user_id: str,
        payload: dict[str, Any],
        max_count: int = -1,
    ) -> dict:
        """
        Star a new response. Returns the created document.

        If `max_count > 0`, enforces a per-user cap by silently evicting the
        oldest entries once the user is at the cap. `max_count == -1` means
        unlimited.
        """
        now = datetime.now(timezone.utc)
        doc = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "workspace_id": payload["workspace_id"],
            "workspace_label": payload.get("workspace_label", ""),
            "query": payload["query"],
            "answer": payload["answer"],
            "sources": payload.get("sources"),
            "starred_at": now,
        }

        # Cap the per-user count only if the plan has a finite limit
        if max_count > 0:
            count = await self.collection.count_documents({"user_id": user_id})
            if count >= max_count:
                to_evict = count - max_count + 1
                oldest_cursor = (
                    self.collection.find({"user_id": user_id}, {"id": 1})
                    .sort("starred_at", 1)
                    .limit(to_evict)
                )
                ids_to_evict: list[str] = [d["id"] async for d in oldest_cursor]
                if ids_to_evict:
                    await self.collection.delete_many(
                        {"user_id": user_id, "id": {"$in": ids_to_evict}}
                    )

        await self.collection.insert_one(doc)
        logger.debug(f"Starred response {doc['id']} for user {user_id}")
        return self._to_public(doc)

    async def list_for_user(self, user_id: str) -> list[dict]:
        """Return all starred responses for a user, newest first."""
        cursor = self.collection.find({"user_id": user_id}, {"_id": 0}).sort("starred_at", -1)
        return [self._to_public(doc) async for doc in cursor]

    async def delete(self, user_id: str, starred_id: str) -> bool:
        """Unstar a specific response. Returns True if something was removed."""
        result = await self.collection.delete_one({"user_id": user_id, "id": starred_id})
        return result.deleted_count > 0

    async def delete_all_for_user(self, user_id: str) -> int:
        """Remove all starred responses for a user. Returns count deleted."""
        result = await self.collection.delete_many({"user_id": user_id})
        return result.deleted_count

    async def find_for_query(
        self, user_id: str, workspace_id: str, query: str
    ) -> Optional[dict]:
        """
        Find an existing starred response matching workspace + query.
        Used by the UI to decide whether the star button is already filled.
        """
        doc = await self.collection.find_one(
            {"user_id": user_id, "workspace_id": workspace_id, "query": query},
            {"_id": 0},
        )
        return self._to_public(doc) if doc else None

    @staticmethod
    def _to_public(doc: dict) -> dict:
        """Strip internal fields."""
        return {k: v for k, v in doc.items() if k not in ("_id", "user_id")}
