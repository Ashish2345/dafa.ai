"""
Active session management — list the devices currently signed in as the
authenticated user, and revoke individual sessions or every other session.

A "session" here is a single refresh-token family: when the user signs in on
a new device a row is inserted into `user_sessions`; subsequent calls to
`/auth/refresh` update `last_seen_at` on that same row rather than creating
a new one. The `sid` JWT claim ties an access token back to its session so
we can mark "This device" on the settings page.

Schema (MongoDB):
    user_sessions {
        session_id: uuid4 str,             # primary key
        user_id: str,
        user_agent: str,
        ip: str,
        country: str | None,               # best-effort GeoIP later
        device_label: str,                 # "Chrome on Windows · Kathmandu"
        created_at: datetime,
        last_seen_at: datetime,
        revoked: bool,
        revoked_at: datetime | None,
    }
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger

from app.db.mongodb import get_database
from app.utils.auth import get_current_user

router = APIRouter(prefix="/user", tags=["user"])


# ── Helpers ─────────────────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    """Best-effort client IP. Prefers X-Forwarded-For, falls back to the raw peer."""
    header = request.headers.get("x-forwarded-for")
    if header:
        return header.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _label_from_user_agent(user_agent: str) -> str:
    """
    Derive a compact "Chrome on Windows" style label from the UA string.
    Good enough for a settings list; we don't need a full UA parser.
    """
    ua = user_agent or ""
    # Browser
    if "Edg/" in ua:
        browser = "Edge"
    elif "OPR/" in ua or "Opera" in ua:
        browser = "Opera"
    elif "Chrome/" in ua and "Chromium" not in ua:
        browser = "Chrome"
    elif "Firefox/" in ua:
        browser = "Firefox"
    elif "Safari/" in ua and "Chrome/" not in ua:
        browser = "Safari"
    else:
        browser = "Unknown browser"

    # OS
    if "iPhone" in ua or "iPad" in ua or "iOS" in ua:
        os = "iOS"
    elif "Android" in ua:
        os = "Android"
    elif "Windows" in ua:
        os = "Windows"
    elif "Mac OS X" in ua or "Macintosh" in ua:
        os = "macOS"
    elif "Linux" in ua:
        os = "Linux"
    else:
        os = "Unknown OS"

    return f"{browser} on {os}"


def _is_mobile(user_agent: str) -> bool:
    return bool(re.search(r"iPhone|iPad|Android|Mobile", user_agent or "", re.I))


async def record_session_on_login(
    db,
    user_id: str,
    request: Request,
) -> str:
    """
    Insert a new session row at login and return the fresh session_id so the
    caller can embed it in the refresh token JWT payload (`sid` claim).
    """
    session_id = str(uuid.uuid4())
    ua = request.headers.get("user-agent", "") if request else ""
    ip = _client_ip(request) if request else "unknown"
    now = datetime.now(timezone.utc)

    await db.user_sessions.insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "user_agent": ua,
        "ip": ip,
        "country": None,
        "device_label": _label_from_user_agent(ua),
        "is_mobile": _is_mobile(ua),
        "created_at": now,
        "last_seen_at": now,
        "revoked": False,
        "revoked_at": None,
    })
    return session_id


async def touch_session_on_refresh(db, session_id: str) -> None:
    """Bump last_seen_at on the session matching this refresh token."""
    if not session_id:
        return
    await db.user_sessions.update_one(
        {"session_id": session_id},
        {"$set": {"last_seen_at": datetime.now(timezone.utc)}},
    )


async def is_session_revoked(db, session_id: str) -> bool:
    """
    True if the session was revoked out-of-band. Used by /auth/refresh to
    block refresh-token reuse after a remote logout.
    """
    if not session_id:
        return False
    row = await db.user_sessions.find_one(
        {"session_id": session_id}, {"_id": 0, "revoked": 1}
    )
    return bool(row and row.get("revoked"))


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/sessions")
async def list_sessions(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return the active sessions for the current user, current one marked."""
    user_id = current_user["sub"]
    current_sid = current_user.get("sid")

    cursor = db.user_sessions.find(
        {"user_id": user_id, "revoked": {"$ne": True}},
    ).sort("last_seen_at", -1)

    items: list[dict[str, Any]] = []
    async for row in cursor:
        row.pop("_id", None)
        items.append({
            "session_id": row["session_id"],
            "device_label": row.get("device_label", "Unknown device"),
            "is_mobile": row.get("is_mobile", False),
            "user_agent": row.get("user_agent", ""),
            "ip": row.get("ip", ""),
            "country": row.get("country"),
            "created_at": _iso(row.get("created_at")),
            "last_seen_at": _iso(row.get("last_seen_at")),
            "is_current": row["session_id"] == current_sid,
        })

    return {"sessions": items}


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Revoke a specific session. Revoking the current session effectively logs
    the user out on the next refresh cycle (the access token still works
    until it expires, but no new tokens will be issued).
    """
    user_id = current_user["sub"]
    result = await db.user_sessions.update_one(
        {"session_id": session_id, "user_id": user_id},
        {"$set": {"revoked": True, "revoked_at": datetime.now(timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    logger.info(f"Session revoked: user={user_id} sid={session_id}")
    return None


@router.post("/sessions/revoke-all-other")
async def revoke_other_sessions(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Revoke every session except the current one."""
    user_id = current_user["sub"]
    current_sid = current_user.get("sid")
    result = await db.user_sessions.update_many(
        {
            "user_id": user_id,
            "revoked": {"$ne": True},
            **({"session_id": {"$ne": current_sid}} if current_sid else {}),
        },
        {"$set": {"revoked": True, "revoked_at": datetime.now(timezone.utc)}},
    )
    logger.info(f"Revoked {result.modified_count} other sessions for user {user_id}")
    return {"revoked": result.modified_count}


# ── Serialization helpers ──────────────────────────────────────────────────

def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
