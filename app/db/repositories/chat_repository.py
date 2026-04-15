"""
Chat repository — conversations sidebar + per-workspace message storage.

Two collections:
- `chat_conversations` — one doc per (user, workspace), for the sidebar list
- `chat_messages` — one doc per (user, workspace), holds the messages array
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


MAX_CONVERSATIONS_PER_USER = 15


def _title_from_query(query: str) -> str:
    t = " ".join(query.strip().split())
    return t if len(t) <= 56 else t[:53] + "\u2026"


def _strip_markup(text: str) -> str:
    import re

    t = re.sub(r"<[^>]+>", "", text)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"#{1,6}\s", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


class ChatRepository:
    """CRUD for chat conversations & messages."""

    def __init__(self, database):
        self.conversations = database.chat_conversations
        self.messages = database.chat_messages

    # ── Conversations ──────────────────────────────────────────────

    async def list_conversations(
        self,
        user_id: str,
        max_age_days: int = -1,
    ) -> list[dict]:
        """
        Return this user's recent conversations, newest first.

        If `max_age_days > 0`, only conversations with activity within that window
        are returned (used to enforce plan's chat_history_days retention).
        """
        from datetime import timedelta

        filter_doc: dict = {"user_id": user_id}
        if max_age_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            filter_doc["last_activity_at"] = {"$gte": cutoff}

        cursor = (
            self.conversations.find(filter_doc, {"_id": 0, "user_id": 0})
            .sort("last_activity_at", -1)
            .limit(MAX_CONVERSATIONS_PER_USER)
        )
        return [doc async for doc in cursor]

    async def record_turn(
        self,
        user_id: str,
        workspace_id: str,
        workspace_label: str,
        query: str,
        answer_preview: str,
    ) -> dict:
        """
        Upsert a conversation entry after a completed Q&A.
        Updates title/preview/last_activity if the conversation already exists.
        """
        now = datetime.now(timezone.utc)
        title = _title_from_query(query)
        preview = _strip_markup(answer_preview)[:120].strip()

        doc = {
            "workspace_id": workspace_id,
            "workspace_label": workspace_label,
            "title": title,
            "preview": preview,
            "last_activity_at": now,
        }

        await self.conversations.update_one(
            {"user_id": user_id, "workspace_id": workspace_id},
            {
                "$set": doc,
                "$setOnInsert": {
                    "user_id": user_id,
                    "id": str(uuid.uuid4()),
                    "created_at": now,
                },
            },
            upsert=True,
        )

        # Prune if the user is over the cap
        count = await self.conversations.count_documents({"user_id": user_id})
        if count > MAX_CONVERSATIONS_PER_USER:
            over = count - MAX_CONVERSATIONS_PER_USER
            old_cursor = (
                self.conversations.find(
                    {"user_id": user_id}, {"id": 1, "workspace_id": 1}
                )
                .sort("last_activity_at", 1)
                .limit(over)
            )
            old_docs: list[dict] = [d async for d in old_cursor]
            old_ids = [d["id"] for d in old_docs]
            old_workspaces = [d["workspace_id"] for d in old_docs if d.get("workspace_id")]
            if old_ids:
                await self.conversations.delete_many(
                    {"user_id": user_id, "id": {"$in": old_ids}}
                )
            # Also delete the corresponding message docs to avoid orphan growth
            if old_workspaces:
                await self.messages.delete_many(
                    {"user_id": user_id, "workspace_id": {"$in": old_workspaces}}
                )
                logger.debug(
                    f"Pruned {len(old_workspaces)} old conversations + messages for user {user_id}"
                )

        result = await self.conversations.find_one(
            {"user_id": user_id, "workspace_id": workspace_id},
            {"_id": 0, "user_id": 0},
        )
        return result or {}

    async def delete_conversation(self, user_id: str, workspace_id: str) -> bool:
        """Remove a conversation + its messages."""
        conv_result = await self.conversations.delete_one(
            {"user_id": user_id, "workspace_id": workspace_id}
        )
        await self.messages.delete_one(
            {"user_id": user_id, "workspace_id": workspace_id}
        )
        return conv_result.deleted_count > 0

    async def clear_all_for_user(self, user_id: str) -> int:
        """Wipe all conversations + messages for a user."""
        r1 = await self.conversations.delete_many({"user_id": user_id})
        r2 = await self.messages.delete_many({"user_id": user_id})
        return r1.deleted_count + r2.deleted_count

    # ── Messages ──────────────────────────────────────────────────

    async def get_messages(self, user_id: str, workspace_id: str) -> list[dict]:
        """Return the messages array for a (user, workspace)."""
        doc = await self.messages.find_one(
            {"user_id": user_id, "workspace_id": workspace_id},
            {"_id": 0, "messages": 1},
        )
        if not doc:
            return []
        return doc.get("messages", [])

    async def save_messages(
        self,
        user_id: str,
        workspace_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Replace the entire messages array for a (user, workspace)."""
        now = datetime.now(timezone.utc)
        await self.messages.update_one(
            {"user_id": user_id, "workspace_id": workspace_id},
            {
                "$set": {"messages": messages, "updated_at": now},
                "$setOnInsert": {
                    "user_id": user_id,
                    "workspace_id": workspace_id,
                    "created_at": now,
                },
            },
            upsert=True,
        )
        logger.debug(
            f"Saved {len(messages)} messages for user {user_id} in {workspace_id}"
        )
