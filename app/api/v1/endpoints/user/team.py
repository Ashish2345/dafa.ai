"""
Team + invites endpoints — Phase 14.

Two new MongoDB collections:
- ``team_memberships``: (owner_user_id, member_user_id, role, joined_at)
    The owner themselves is implicit: the API synthesises them into the /team
    response so there's always exactly one "Owner" row for the caller.
- ``team_invites``: (id, owner_user_id, email, role, scope, note, token,
    expires_at, status, created_at)
    Email sending is deferred — for now we create the invite record and return
    the data so the frontend can still surface "Invited 2h ago" UI.

All endpoints are scoped to the *current user's* team. The current user is
always the owner in this simple model; shared ownership / invite-acceptance
is Phase 15.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, EmailStr, Field

from app.db.mongodb import get_database
from app.db.repositories.user_repository import UserRepository
from app.services.email_service import EmailService, email_service_dep
from app.settings import settings as app_settings
from app.utils.auth import get_current_user


router = APIRouter(prefix="/user", tags=["user"])

# ── Constants ───────────────────────────────────────────────────────────────

INVITE_TTL_DAYS = 7

TeamRole = Literal["owner", "admin", "member", "viewer"]
InviteScope = Literal["workspace-pinned", "pinned-only", "none"]
InviteStatus = Literal["pending", "accepted", "revoked"]


# ── Pydantic schemas ────────────────────────────────────────────────────────


class TeamMember(BaseModel):
    id: str                      # == member_user_id when joined, else invite id for pending
    user_id: Optional[str] = None
    email: str
    full_name: Optional[str] = None
    role: TeamRole
    is_you: bool = False
    joined_at: Optional[datetime] = None


class TeamInvite(BaseModel):
    id: str
    owner_user_id: str
    email: str
    role: TeamRole
    scope: InviteScope
    note: Optional[str] = None
    token: str
    expires_at: datetime
    status: InviteStatus
    created_at: datetime


class TeamResponse(BaseModel):
    members: list[TeamMember]
    pending_invites: list[TeamInvite]


class CreateInvitesRequest(BaseModel):
    emails: list[EmailStr] = Field(..., min_length=1, max_length=50)
    role: TeamRole = Field(default="member")
    scope: InviteScope = Field(default="workspace-pinned")
    note: Optional[str] = Field(default=None, max_length=500)


class CreateInvitesResponse(BaseModel):
    created: list[TeamInvite]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _new_invite_doc(
    owner_user_id: str,
    email: str,
    role: TeamRole,
    scope: InviteScope,
    note: Optional[str],
) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "id": str(uuid.uuid4()),
        "owner_user_id": owner_user_id,
        "email": email,
        "role": role,
        "scope": scope,
        "note": note,
        "token": secrets.token_urlsafe(24),
        "expires_at": now + timedelta(days=INVITE_TTL_DAYS),
        "status": "pending",
        "created_at": now,
    }


def _strip_id(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "_id"}


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/team", response_model=TeamResponse)
async def get_team(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Return the caller's team — themselves as owner, any joined members, and any
    pending invites they've sent.
    """
    owner_id = current_user["sub"]

    # Owner (the caller) — always the first row.
    user_repo = UserRepository(db)
    me = await user_repo.get_by_id(owner_id)
    owner_row = TeamMember(
        id=owner_id,
        user_id=owner_id,
        email=(me or {}).get("email") or current_user.get("email") or "",
        full_name=(me or {}).get("full_name"),
        role="owner",
        is_you=True,
        joined_at=(me or {}).get("created_at"),
    )

    members: list[TeamMember] = [owner_row]

    memberships_cursor = db.team_memberships.find({"owner_user_id": owner_id})
    async for m in memberships_cursor:
        muser = await user_repo.get_by_id(m.get("member_user_id"))
        if not muser:
            continue
        members.append(
            TeamMember(
                id=m.get("member_user_id"),
                user_id=m.get("member_user_id"),
                email=muser.get("email", ""),
                full_name=muser.get("full_name"),
                role=m.get("role", "member"),
                is_you=False,
                joined_at=m.get("joined_at"),
            )
        )

    pending: list[TeamInvite] = []
    invites_cursor = db.team_invites.find(
        {"owner_user_id": owner_id, "status": "pending"}
    ).sort("created_at", -1)
    async for inv in invites_cursor:
        pending.append(TeamInvite(**_strip_id(inv)))

    return TeamResponse(members=members, pending_invites=pending)


@router.post("/invites", response_model=CreateInvitesResponse, status_code=status.HTTP_201_CREATED)
async def create_invites(
    body: CreateInvitesRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
    email_service: EmailService = Depends(email_service_dep),
):
    """
    Create one invite record per email. De-duplicates against existing *pending*
    invites for the same (owner, email) — returns the existing row instead of
    creating a duplicate.

    Phase 16d: after the row is persisted we fire an invite email via the
    configured email service. A failed send does NOT abort the request — the
    invite row is still returned so the owner can hit "Resend" from the UI.
    """
    owner_id = current_user["sub"]
    created: list[TeamInvite] = []

    # Load owner profile once so every invite email has a consistent sender
    # display name + team label.
    user_repo = UserRepository(db)
    owner_profile = await user_repo.get_by_id(owner_id)
    owner_name = (owner_profile or {}).get("full_name") or current_user.get("email") or "A MeroDafa user"
    team_name = _derive_team_name(owner_profile, owner_id)

    for email in body.emails:
        email_str = str(email)
        existing = await db.team_invites.find_one(
            {"owner_user_id": owner_id, "email": email_str, "status": "pending"}
        )
        if existing:
            created.append(TeamInvite(**_strip_id(existing)))
            continue

        doc = _new_invite_doc(owner_id, email_str, body.role, body.scope, body.note)
        await db.team_invites.insert_one(doc)
        created.append(TeamInvite(**_strip_id(doc)))

        # Send email best-effort; failure only logs.
        try:
            await email_service.send_invite(
                to=email_str,
                token=doc["token"],
                owner_name=owner_name,
                team_name=team_name,
                note=doc.get("note"),
                role=doc["role"],
                frontend_url=app_settings.frontend_url,
            )
        except Exception as exc:
            logger.warning(f"Invite email to {email_str} failed: {exc}")

    logger.info(f"User {owner_id} created {len(created)} invite(s)")
    return CreateInvitesResponse(created=created)


def _derive_team_name(owner_profile: Optional[dict], owner_id: str) -> str:
    """
    Best-effort label for the team. Preferences have `organization`; fall
    back to the owner's name; finally to a generic "MeroDafa team".
    """
    if owner_profile:
        org = owner_profile.get("organization")
        if org:
            return str(org)
        full = owner_profile.get("full_name")
        if full:
            return f"{full}'s team"
    return "a MeroDafa team"


@router.post("/invites/{invite_id}/resend", response_model=TeamInvite)
async def resend_invite(
    invite_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
    email_service: EmailService = Depends(email_service_dep),
):
    """
    Push the invite's ``expires_at`` out by another 7 days, re-send the
    invite email, and return the updated row.
    """
    owner_id = current_user["sub"]

    existing = await db.team_invites.find_one(
        {"id": invite_id, "owner_user_id": owner_id}
    )
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found")

    if existing.get("status") != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only pending invites can be resent",
        )

    new_expiry = datetime.now(timezone.utc) + timedelta(days=INVITE_TTL_DAYS)
    await db.team_invites.update_one(
        {"id": invite_id, "owner_user_id": owner_id},
        {"$set": {"expires_at": new_expiry}},
    )
    existing["expires_at"] = new_expiry

    # Phase 16d: re-fire the invite email. Best-effort — caller still gets
    # the updated invite row on failure.
    try:
        user_repo = UserRepository(db)
        owner_profile = await user_repo.get_by_id(owner_id)
        owner_name = (owner_profile or {}).get("full_name") or current_user.get("email") or "A MeroDafa user"
        await email_service.send_invite(
            to=existing["email"],
            token=existing["token"],
            owner_name=owner_name,
            team_name=_derive_team_name(owner_profile, owner_id),
            note=existing.get("note"),
            role=existing.get("role", "member"),
            frontend_url=app_settings.frontend_url,
        )
    except Exception as exc:
        logger.warning(f"Invite resend email to {existing['email']} failed: {exc}")

    return TeamInvite(**_strip_id(existing))


@router.delete("/invites/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_invite(
    invite_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Mark an invite as revoked. Idempotent — returns 204 even if already revoked."""
    owner_id = current_user["sub"]
    result = await db.team_invites.update_one(
        {"id": invite_id, "owner_user_id": owner_id},
        {"$set": {"status": "revoked"}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invite not found")
    return None


@router.delete("/team/{member_user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    member_user_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Owner-only: remove a joined member from the caller's team."""
    owner_id = current_user["sub"]
    if member_user_id == owner_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove yourself — transfer ownership first",
        )

    result = await db.team_memberships.delete_one(
        {"owner_user_id": owner_id, "member_user_id": member_user_id}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")
    return None
