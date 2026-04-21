"""
Public invite redemption endpoints — Phase 16d.

Two routes, both unauthenticated:

    GET  /invite/info?token=<...>    – peek at a pending invite so the accept
                                       page can render the team + role + note
                                       without knowing the token-holder yet.
    POST /invite/accept              – redeem a token. Two code paths:
                                        * existing user → link to team, return
                                          fresh tokens (logs them in).
                                        * brand-new email → require ``password``
                                          in the body, create a verified user,
                                          link to team, return fresh tokens.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from loguru import logger
from pydantic import BaseModel, Field

from app.db.mongodb import get_database
from app.db.repositories.user_repository import UserRepository
from app.utils.auth import (
    create_access_token,
    create_refresh_token,
    hash_password,
)


router = APIRouter(prefix="/invite", tags=["auth"])


# ── Shapes ─────────────────────────────────────────────────────────────────

class InviteInfo(BaseModel):
    email: str
    role: str
    scope: str
    note: Optional[str] = None
    owner_name: str
    team_name: str
    expires_at: datetime
    status: str
    is_existing_user: bool = Field(
        ..., description="True when the invite email matches an existing user."
    )


class AcceptInviteBody(BaseModel):
    token: str = Field(..., min_length=1)
    password: Optional[str] = Field(
        default=None,
        min_length=8,
        description="Required only when the invite email is new to MeroDafa.",
    )
    full_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional display name for brand-new users.",
    )


class AcceptInviteResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    is_new_user: bool


# ── Helpers ────────────────────────────────────────────────────────────────

async def _fetch_invite(db, token: str) -> dict:
    invite = await db.team_invites.find_one({"token": token})
    if not invite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "invalid", "message": "This invite doesn't exist."},
        )
    return invite


def _validate_status(invite: dict) -> None:
    """Reject invites that have been accepted, revoked, or expired."""
    if invite.get("status") == "accepted":
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail={"error": "accepted", "message": "This invite has already been used."},
        )
    if invite.get("status") == "revoked":
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail={"error": "revoked", "message": "This invite was revoked by the owner."},
        )
    exp = invite.get("expires_at")
    if isinstance(exp, datetime):
        # Normalize tz-naive records (pre-tz-aware inserts) to UTC.
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail={"error": "expired", "message": "This invite has expired."},
            )


async def _owner_context(db, owner_id: str) -> tuple[str, str]:
    user_repo = UserRepository(db)
    owner = await user_repo.get_by_id(owner_id)
    if not owner:
        return "MeroDafa owner", "a MeroDafa team"
    name = owner.get("full_name") or owner.get("email") or "MeroDafa owner"
    org = owner.get("organization")
    team = str(org) if org else (f"{name}'s team" if name else "a MeroDafa team")
    return name, team


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/info", response_model=InviteInfo)
async def invite_info(
    token: str = Query(..., min_length=1),
    db=Depends(get_database),
):
    """Public lookup — lets the accept page pre-render before login."""
    invite = await _fetch_invite(db, token)
    _validate_status(invite)

    owner_name, team_name = await _owner_context(db, invite["owner_user_id"])

    user_repo = UserRepository(db)
    existing = await user_repo.get_by_email(invite["email"])

    return InviteInfo(
        email=invite["email"],
        role=invite["role"],
        scope=invite["scope"],
        note=invite.get("note"),
        owner_name=owner_name,
        team_name=team_name,
        expires_at=invite["expires_at"],
        status=invite.get("status", "pending"),
        is_existing_user=bool(existing),
    )


@router.post("/accept", response_model=AcceptInviteResponse)
async def accept_invite(
    body: AcceptInviteBody,
    request: Request,
    db=Depends(get_database),
):
    """
    Redeem an invite token. Two shapes depending on whether the invite email
    matches an existing user:

    * **Existing user** — body only needs ``token``. We link them to the team
      (idempotent — no-op if the row already exists) and issue tokens.

    * **New user** — body must include ``password`` (min 8 chars). We create
      a verified user record, link them to the team, and issue tokens.

    The caller is logged in on success — the frontend stores the returned
    tokens in AuthContext just like a normal login.
    """
    from app.api.v1.endpoints.user.sessions import record_session_on_login

    invite = await _fetch_invite(db, body.token)
    _validate_status(invite)

    user_repo = UserRepository(db)
    existing = await user_repo.get_by_email(invite["email"])

    if existing:
        user = existing
        is_new_user = False
    else:
        if not body.password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "password_required",
                    "message": "This email isn't registered yet — set a password to create your account.",
                },
            )
        try:
            created = await user_repo.create_user(
                email=invite["email"],
                hashed_password=hash_password(body.password),
                full_name=body.full_name or invite["email"].split("@")[0],
            )
        except ValueError as exc:
            # Race: another caller registered the email between the lookup
            # above and now. Re-fetch and fall through to the existing-user
            # path.
            logger.warning(f"Accept invite race on create_user: {exc}")
            created_user = await user_repo.get_by_email(invite["email"])
            if not created_user:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create user account",
                ) from exc
            user = created_user
            is_new_user = False
        else:
            # Auto-verify — the token proves they own the email.
            await user_repo.set_verified(invite["email"])
            user = await user_repo.get_by_email(invite["email"]) or created
            is_new_user = True

    # Link to the team (idempotent via upsert).
    now = datetime.now(timezone.utc)
    await db.team_memberships.update_one(
        {
            "owner_user_id": invite["owner_user_id"],
            "member_user_id": user["user_id"],
        },
        {
            "$setOnInsert": {
                "owner_user_id": invite["owner_user_id"],
                "member_user_id": user["user_id"],
                "role": invite["role"],
                "scope": invite["scope"],
                "joined_at": now,
            }
        },
        upsert=True,
    )

    # Mark invite accepted.
    await db.team_invites.update_one(
        {"token": body.token},
        {"$set": {"status": "accepted", "accepted_at": now, "accepted_by": user["user_id"]}},
    )

    # Session-tracked login, matching /auth/login.
    session_id = await record_session_on_login(db, user["user_id"], request)
    access_token = create_access_token(
        user_id=user["user_id"], email=user["email"], role=user.get("role", "user"),
        session_id=session_id,
    )
    refresh_token = create_refresh_token(user_id=user["user_id"], session_id=session_id)

    logger.info(
        f"Invite accepted: user={user['user_id']} owner={invite['owner_user_id']} "
        f"new={is_new_user}"
    )
    return AcceptInviteResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        is_new_user=is_new_user,
    )
