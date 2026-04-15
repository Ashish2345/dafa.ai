"""
Feedback repository — stores user-submitted feedback + optional screenshot as base64.

Design note: screenshots are stored as base64 inside the same Mongo document for
simplicity. Max ~3MB per image keeps us well under the 16MB doc limit. If usage
grows we can migrate images to disk/S3 later.
"""

import base64
import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger


MAX_IMAGE_BYTES = 3 * 1024 * 1024  # 3 MB raw (≈ 4 MB base64)


class FeedbackRepository:
    """CRUD for the `feedback` collection."""

    def __init__(self, database):
        self.collection = database.feedback

    async def create(
        self,
        user_id: str,
        email: str,
        category: str,
        message: str,
        image_bytes: Optional[bytes] = None,
        image_content_type: Optional[str] = None,
    ) -> dict:
        """Store a feedback entry. Returns the created document (without image bytes)."""
        now = datetime.now(timezone.utc)

        doc = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "email": email,
            "category": category,
            "message": message,
            "status": "new",
            "created_at": now,
            "has_image": False,
        }

        if image_bytes:
            if len(image_bytes) > MAX_IMAGE_BYTES:
                raise ValueError(
                    f"Image too large ({len(image_bytes)} bytes). "
                    f"Maximum is {MAX_IMAGE_BYTES} bytes (3 MB)."
                )
            doc["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
            doc["image_content_type"] = image_content_type or "image/png"
            doc["image_size_bytes"] = len(image_bytes)
            doc["has_image"] = True

        await self.collection.insert_one(doc)
        logger.info(
            f"Feedback {doc['id']} received from {email} "
            f"({category}, {'with' if image_bytes else 'no'} image)"
        )
        return self._to_public(doc)

    @staticmethod
    def _to_public(doc: dict) -> dict:
        """Strip Mongo _id and heavy image bytes from response."""
        return {
            k: v
            for k, v in doc.items()
            if k not in ("_id", "image_base64", "user_id")
        }
