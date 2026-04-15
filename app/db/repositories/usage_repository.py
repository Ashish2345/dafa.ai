"""
Usage repository — daily per-user action counters.

Schema for `usage_counters` docs:
    {
      user_id: str,
      date: "2026-04-14",                       # YYYY-MM-DD in UTC
      counts: { search: int, pdf_download: int, ... },
      updated_at: datetime,
    }

Index: (user_id, date) unique — one doc per user per day.
"""

from datetime import datetime, timezone
from typing import Any

from loguru import logger


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class UsageRepository:
    """CRUD for the `usage_counters` MongoDB collection."""

    def __init__(self, database):
        self.collection = database.usage_counters

    async def get_today(self, user_id: str) -> dict[str, int]:
        """
        Return this user's action counts for the current UTC day.
        Returns an empty dict if nothing recorded yet.
        """
        today = _today_utc()
        doc = await self.collection.find_one(
            {"user_id": user_id, "date": today},
            {"_id": 0, "counts": 1},
        )
        return (doc or {}).get("counts", {})

    async def increment(self, user_id: str, action: str) -> int:
        """
        Atomically increment `counts.<action>` for today. Upserts the daily doc.
        Returns the NEW count for this action today.
        """
        today = _today_utc()
        now = datetime.now(timezone.utc)
        field = f"counts.{action}"

        result = await self.collection.find_one_and_update(
            {"user_id": user_id, "date": today},
            {
                "$inc": {field: 1},
                "$set": {"updated_at": now},
                "$setOnInsert": {
                    "user_id": user_id,
                    "date": today,
                    "created_at": now,
                },
            },
            upsert=True,
            return_document=True,
        )
        new_count = (result or {}).get("counts", {}).get(action, 1)
        logger.debug(f"Usage {user_id} {today} {action} → {new_count}")
        return new_count

    async def check_quota(
        self,
        user_id: str,
        action: str,
        limit: int,
    ) -> tuple[bool, int, int]:
        """
        Check whether the user has quota left for this action.
        `limit == -1` means unlimited → always allowed.

        Returns:
            (allowed, used_so_far, remaining)
            `remaining` is -1 if unlimited.
        """
        if limit == -1:
            used = (await self.get_today(user_id)).get(action, 0)
            return True, used, -1

        used = (await self.get_today(user_id)).get(action, 0)
        remaining = max(0, limit - used)
        return used < limit, used, remaining

    @staticmethod
    def today_utc() -> str:
        """Public accessor so endpoints can return the same date string."""
        return _today_utc()

    async def clear_for_user(self, user_id: str) -> int:
        """Wipe all usage records for a user (for account deletion flows)."""
        result = await self.collection.delete_many({"user_id": user_id})
        return result.deleted_count

    async def get_last_n_days(self, user_id: str, days: int = 7) -> list[dict]:
        """
        Return an array of daily usage for the last N days (oldest first).
        Missing days are filled with zero counts.
        """
        from datetime import timedelta

        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days - 1)
        start_str = start.isoformat()
        end_str = end.isoformat()

        cursor = self.collection.find(
            {
                "user_id": user_id,
                "date": {"$gte": start_str, "$lte": end_str},
            },
            {"_id": 0, "date": 1, "counts": 1},
        )

        by_date: dict[str, dict[str, int]] = {}
        async for doc in cursor:
            by_date[doc["date"]] = doc.get("counts", {})

        out: list[dict] = []
        for i in range(days):
            d = (start + timedelta(days=i)).isoformat()
            counts = by_date.get(d, {})
            out.append(
                {
                    "date": d,
                    "searches": counts.get("search", 0),
                    "pdf_downloads": counts.get("pdf_download", 0),
                }
            )
        return out
