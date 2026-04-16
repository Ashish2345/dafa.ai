"""
Verification code repository — stores and validates 6-digit OTP codes.

Used for email verification (signup) and password reset.
Codes are hashed with bcrypt so a DB leak doesn't expose them.
Each (email, purpose) pair has at most one active code at a time.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt as _bcrypt
from loguru import logger

from app.settings import settings


def _generate_code() -> str:
    """Generate a cryptographically random 6-digit numeric code."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_code(code: str) -> str:
    return _bcrypt.hashpw(code.encode(), _bcrypt.gensalt()).decode()


def _verify_code(code: str, hashed: str) -> bool:
    return _bcrypt.checkpw(code.encode(), hashed.encode())


class VerificationRepository:
    """CRUD for the `verification_codes` MongoDB collection."""

    def __init__(self, database):
        self.collection = database.verification_codes

    async def create_code(self, email: str, purpose: str = "email_verify") -> str:
        """
        Generate a 6-digit code, store (hashed) for the given email + purpose,
        and return the plain code so it can be emailed.

        Replaces any previous code for the same (email, purpose).
        """
        code = _generate_code()
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=settings.verification_code_expire_minutes)

        await self.collection.update_one(
            {"email": email, "purpose": purpose},
            {
                "$set": {
                    "hashed_code": _hash_code(code),
                    "attempts": 0,
                    "created_at": now,
                    "expires_at": expires_at,
                },
                "$setOnInsert": {"email": email, "purpose": purpose},
            },
            upsert=True,
        )
        logger.debug(f"Verification code created: {email} / {purpose}")
        return code

    async def verify(self, email: str, code: str, purpose: str = "email_verify") -> bool:
        """
        Validate a code. Returns True on match, False otherwise.

        - Expired codes → False
        - >5 failed attempts → auto-delete (brute-force protection)
        - On success the code doc is deleted (single-use)
        """
        doc = await self.collection.find_one({"email": email, "purpose": purpose})
        if not doc:
            return False

        # Expired? (MongoDB stores naive UTC datetimes)
        expires = doc["expires_at"]
        now = datetime.now(timezone.utc)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires < now:
            await self.collection.delete_one({"_id": doc["_id"]})
            return False

        # Too many attempts?
        if doc.get("attempts", 0) >= 5:
            await self.collection.delete_one({"_id": doc["_id"]})
            logger.warning(f"Verification code locked out: {email} / {purpose}")
            return False

        if _verify_code(code, doc["hashed_code"]):
            # Success — delete the code (single-use)
            await self.collection.delete_one({"_id": doc["_id"]})
            return True

        # Wrong code — increment attempt counter
        await self.collection.update_one(
            {"_id": doc["_id"]},
            {"$inc": {"attempts": 1}},
        )
        return False

    async def delete(self, email: str, purpose: str = "email_verify") -> None:
        """Remove any pending code for an email + purpose."""
        await self.collection.delete_many({"email": email, "purpose": purpose})
