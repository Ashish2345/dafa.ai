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
        """Return word bboxes within a character offset range, grouped by page.

        Uses MongoDB $elemMatch to pre-filter pages that contain at least one
        word in range, reducing data transfer for large documents.
        """
        cursor = self.collection.find(
            {
                "document_id": document_id,
                "words": {
                    "$elemMatch": {
                        "char_offset": {"$gte": start_char, "$lt": end_char}
                    }
                },
            },
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

    async def get_words_in_spatial_region(
        self,
        document_id: str,
        page: int,
        y0: float,
        y2: float,
    ) -> list[dict]:
        """Return words on a specific page whose y-center falls within [y0, y2].

        Coordinates are normalized 0-1. Used when char offsets are unavailable
        but we have the section's approximate spatial region from page_bboxes.
        """
        doc = await self.collection.find_one(
            {"document_id": document_id, "page": page},
            {"_id": 0, "words": 1},
        )
        if not doc:
            return []

        words = []
        for w in doc.get("words", []):
            # Check if word's vertical center is within the region
            word_y_center = (w.get("y0", 0) + w.get("y2", 0)) / 2
            if y0 <= word_y_center <= y2:
                words.append(w)
        return words

    async def delete_document(self, document_id: str) -> None:
        """Delete all page bbox data for a document."""
        await self.collection.delete_many({"document_id": document_id})
