"""
Repository for word-level OCR bounding boxes.

Stores one MongoDB document per page per ingested document.
Used by the highlights endpoint to return precise text regions
for citation highlighting on page images.
"""

from datetime import datetime, timezone


class OcrBboxRepository:
    def __init__(self, database):
        self.collection = database.page_ocr_bboxes

    async def save_page(
        self,
        document_id: str,
        page: int,
        words: list[dict],
    ) -> None:
        """Save word-level bboxes for a single page (upsert)."""
        await self.collection.update_one(
            {"document_id": document_id, "page": page},
            {
                "$set": {
                    "document_id": document_id,
                    "page": page,
                    "words": words,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc),
                },
            },
            upsert=True,
        )

    async def get_words_in_range(
        self,
        document_id: str,
        start_char: int,
        end_char: int,
    ) -> list[dict]:
        """Return word bboxes within a character offset range, grouped by page."""
        cursor = self.collection.find(
            {"document_id": document_id},
            {"_id": 0, "page": 1, "words": 1},
        ).sort("page", 1)

        result = []
        async for doc in cursor:
            filtered_words = [
                w for w in doc.get("words", [])
                if start_char <= w.get("char_offset", -1) < end_char
            ]
            if filtered_words:
                result.append({
                    "page": doc["page"],
                    "words": filtered_words,
                })
        return result

    async def delete_document(self, document_id: str) -> None:
        """Delete all page bbox data for a document."""
        await self.collection.delete_many({"document_id": document_id})
