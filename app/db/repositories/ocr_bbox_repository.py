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
            # Check if word starts within the region (using top edge, not center,
            # so words that begin below the boundary are excluded)
            word_top = w.get("y0", 0)
            if y0 <= word_top <= y2:
                words.append(w)
        return words

    async def get_page_words(
        self,
        document_id: str,
        page: int,
    ) -> list[dict]:
        """Return every word on a single page (for PDF-style text selection overlay).

        Each word has normalized 0-1 coordinates (x0/y0/x2/y2), ``text``, and
        optional ``line``/``block`` metadata used downstream for reading-order
        grouping.
        """
        doc = await self.collection.find_one(
            {"document_id": document_id, "page": page},
            {"_id": 0, "words": 1},
        )
        return doc.get("words", []) if doc else []

    async def get_all_words(
        self,
        document_id: str,
    ) -> dict[int, list[dict]]:
        """Return every word on every page, keyed by page number.

        Used by the frontend to populate the text-selection overlay in a single
        round trip instead of N per-page fetches.
        """
        cursor = self.collection.find(
            {"document_id": document_id},
            {"_id": 0, "page": 1, "words": 1},
        ).sort("page", 1)

        out: dict[int, list[dict]] = {}
        async for doc in cursor:
            out[doc["page"]] = doc.get("words", [])
        return out

    async def delete_document(self, document_id: str) -> None:
        """Delete all page bbox data for a document."""
        await self.collection.delete_many({"document_id": document_id})
