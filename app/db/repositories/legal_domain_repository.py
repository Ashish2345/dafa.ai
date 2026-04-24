"""
Legal-domain catalog repository — Mongo-backed CRUD + idempotent seed.

Collection: `legal_domains`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


class LegalDomainRepository:
    """CRUD for the `legal_domains` catalog."""

    def __init__(self, database):
        self.collection = database.legal_domains

    async def list_all(self) -> list[dict]:
        """Return every domain, sorted by display_order asc then name asc."""
        cursor = (
            self.collection.find({}, {"_id": 0})
            .sort([("display_order", 1), ("name", 1)])
        )
        return [doc async for doc in cursor]

    async def get(self, slug: str) -> Optional[dict]:
        return await self.collection.find_one({"slug": slug}, {"_id": 0})

    async def upsert(self, row: dict[str, Any]) -> None:
        """Idempotent — used by the seed pipeline. Preserves created_at."""
        now = datetime.now(timezone.utc)
        slug = row["slug"]
        await self.collection.update_one(
            {"slug": slug},
            {
                "$set": {**row, "updated_at": now},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def update(self, slug: str, patch: dict[str, Any]) -> Optional[dict]:
        if not patch:
            return await self.get(slug)
        patch["updated_at"] = datetime.now(timezone.utc)
        result = await self.collection.find_one_and_update(
            {"slug": slug},
            {"$set": patch},
            projection={"_id": 0},
            return_document=True,
        )
        return result

    async def delete(self, slug: str) -> bool:
        result = await self.collection.delete_one({"slug": slug})
        return result.deleted_count > 0

    async def ensure_indexes(self) -> None:
        """Unique index on slug — safe to re-run on startup."""
        try:
            await self.collection.create_index("slug", unique=True)
        except Exception as e:  # pragma: no cover — mongo connection lifecycle
            logger.warning(f"Could not create legal_domains.slug index: {e}")
