"""
User repository — MongoDB CRUD operations for the users collection.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from app.config.plan_loader import plan_catalog


class UserRepository:
    """CRUD operations for the `users` MongoDB collection."""

    def __init__(self, database):
        self.collection = database.users

    async def create_user(
        self,
        email: str,
        hashed_password: str,
        full_name: str,
        role: str = "user",
    ) -> dict:
        """
        Insert a new user document.

        Returns:
            The created user document (without hashed_password).

        Raises:
            ValueError: If a user with the same email already exists.
        """
        existing = await self.collection.find_one({"email": email})
        if existing:
            raise ValueError(f"User with email '{email}' already exists")

        now = datetime.now(timezone.utc)
        user_doc = {
            "user_id": str(uuid.uuid4()),
            "email": email,
            "hashed_password": hashed_password,
            "full_name": full_name,
            "role": role,
            "is_active": True,
            "is_verified": False,
            "plan": plan_catalog.default_plan_id,
            "created_at": now,
            "updated_at": now,
        }

        await self.collection.insert_one(user_doc)
        logger.info(f"Created user: {email} (plan: {user_doc['plan']})")

        return self._to_public(user_doc)

    async def get_by_email(self, email: str) -> Optional[dict]:
        """
        Fetch a user by email, including hashed_password (for auth verification).

        Returns None if not found.
        """
        return await self.collection.find_one({"email": email}, {"_id": 0})

    async def get_by_id(self, user_id: str) -> Optional[dict]:
        """
        Fetch a user by user_id (without hashed_password).

        Returns None if not found.
        """
        doc = await self.collection.find_one({"user_id": user_id}, {"_id": 0, "hashed_password": 0})
        return doc

    async def set_verified(self, email: str) -> bool:
        """Mark a user as email-verified. Returns True if updated."""
        result = await self.collection.update_one(
            {"email": email},
            {"$set": {"is_verified": True, "updated_at": datetime.now(timezone.utc)}},
        )
        return result.modified_count > 0

    async def update_password(self, email: str, hashed_password: str) -> bool:
        """Update a user's password. Returns True if updated."""
        result = await self.collection.update_one(
            {"email": email},
            {"$set": {"hashed_password": hashed_password, "updated_at": datetime.now(timezone.utc)}},
        )
        return result.modified_count > 0

    @staticmethod
    def _to_public(user_doc: dict) -> dict:
        """Return user dict without sensitive fields."""
        return {k: v for k, v in user_doc.items() if k not in ("_id", "hashed_password")}
