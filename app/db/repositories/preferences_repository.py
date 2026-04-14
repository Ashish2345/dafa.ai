"""
User preferences repository — MongoDB CRUD for per-user UI/profile settings.

Stores one document per user in the `user_preferences` collection, keyed by user_id.
If a user has no preferences document yet, `get_or_create_defaults` returns defaults
without inserting — insert happens on first real update.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


DEFAULT_PREFERENCES: dict[str, Any] = {
    "organization": "",
    "role": "",
    "theme": "light",
    "language": "en",
    "font_size_px": 15,
    "notify_new_gazettes": True,
    "notify_product_updates": True,
    "notify_weekly_roundup": True,
    "starred_collections": [],
}


class PreferencesRepository:
    """CRUD for the `user_preferences` MongoDB collection."""

    def __init__(self, database):
        self.collection = database.user_preferences

    async def get_for_user(self, user_id: str) -> dict:
        """
        Return the preferences for a user, falling back to defaults if none exist.
        Never raises for missing data — always returns a usable dict.
        """
        doc = await self.collection.find_one({"user_id": user_id}, {"_id": 0})
        if not doc:
            return {**DEFAULT_PREFERENCES, "updated_at": None}
        return {**DEFAULT_PREFERENCES, **doc}

    async def update_for_user(self, user_id: str, patch: dict[str, Any]) -> dict:
        """
        Upsert a preferences document for the user.
        Only applies non-None fields from `patch`.
        Returns the full preferences after the update.
        """
        # Strip None values so callers can send partial updates
        clean_patch = {k: v for k, v in patch.items() if v is not None}

        if not clean_patch:
            # Nothing to update — just return current state
            return await self.get_for_user(user_id)

        now = datetime.now(timezone.utc)
        clean_patch["updated_at"] = now

        await self.collection.update_one(
            {"user_id": user_id},
            {
                "$set": clean_patch,
                "$setOnInsert": {"user_id": user_id, "created_at": now},
            },
            upsert=True,
        )
        logger.debug(f"Updated preferences for user {user_id}: {list(clean_patch.keys())}")

        return await self.get_for_user(user_id)

    async def delete_for_user(self, user_id: str) -> bool:
        """Delete a user's preferences document (used when account is deleted)."""
        result = await self.collection.delete_one({"user_id": user_id})
        return result.deleted_count > 0
