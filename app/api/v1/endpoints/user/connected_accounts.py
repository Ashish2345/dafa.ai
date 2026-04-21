"""
Connected accounts endpoint — surface third-party identity providers linked
to the authenticated user. Phase 16 only reads the flag written by the Google
OAuth callback; future phases will add linking/unlinking flows.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends

from app.db.mongodb import get_database
from app.utils.auth import get_current_user

router = APIRouter(prefix="/user", tags=["user"])


def _iso(value: Any) -> str | None:
    """Coerce a Mongo datetime into ISO-8601; tolerate strings and None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@router.get("/connected-accounts")
async def get_connected_accounts(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Return the list of third-party identity providers linked to the user.

    For Phase 16 we only detect Google — users created or logged in via the
    `/auth/google/callback` endpoint have `google_linked_at` set on their
    user document. If nothing is linked the array is empty.

    TODO-phase17: actually allow disconnecting a provider (currently returning
    the field lets the frontend grey out its Disconnect button).
    """
    user_id = current_user["sub"]
    doc = await db.users.find_one(
        {"user_id": user_id},
        {
            "_id": 0,
            "email": 1,
            "google_linked_at": 1,
        },
    )

    accounts: list[dict[str, Any]] = []
    if doc and doc.get("google_linked_at"):
        accounts.append({
            "provider": "google",
            "email": doc.get("email", ""),
            "connected_at": _iso(doc.get("google_linked_at")),
            "is_primary": True,
        })

    return {"accounts": accounts}
