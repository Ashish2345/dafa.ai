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

    async def save(
        self, document_id: str, filename: str, metadata: dict[str, Any], strategy: str,
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
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def get(self, document_id: str) -> Optional[dict]:
        return await self.collection.find_one({"document_id": document_id}, {"_id": 0})

    async def list_all(self, skip: int = 0, limit: int = 50) -> list[dict]:
        cursor = self.collection.find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit)
        return [doc async for doc in cursor]

    async def delete(self, document_id: str) -> bool:
        result = await self.collection.delete_one({"document_id": document_id})
        return result.deleted_count > 0
