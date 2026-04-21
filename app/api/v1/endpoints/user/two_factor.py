"""
Two-factor authentication endpoints — TOTP (RFC 6238) setup, verification,
disable, and backup-code regeneration.

Flow:
    1. POST /user/2fa/setup        → generate secret + QR + 10 backup codes
                                     (stored as totp_secret_pending; not yet active)
    2. POST /user/2fa/verify       → user scans QR, submits a 6-digit code;
                                     on success the secret graduates to
                                     totp_secret + totp_enabled=True, and
                                     hashed backup codes are persisted.
    3. POST /user/2fa/disable      → requires a valid TOTP or backup code;
                                     clears all 2FA fields.
    4. POST /user/2fa/regenerate-backup-codes → rotates the backup-code set.

Storage:
    On each user document:
      - totp_secret_pending: str | None        (pre-verify holding slot)
      - totp_secret: str | None                (active secret)
      - totp_enabled: bool                     (gate checked at login)
      - totp_backup_codes: list[str]           (bcrypt-hashed)
      - totp_enrolled_at: datetime | None      (telemetry only)

Backup codes are shown once — only hashes are stored.
"""

import base64
import io
import secrets as secrets_module

import pyotp
import qrcode
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from app.db.mongodb import get_database
from app.utils.auth import get_current_user, hash_password, verify_password

router = APIRouter(prefix="/user/2fa", tags=["user"])


# ── Shapes ──────────────────────────────────────────────────────────────────

class TwoFactorSetupResponse(BaseModel):
    secret: str = Field(..., description="Base32 TOTP secret to type manually if QR scan fails")
    otpauth_url: str = Field(..., description="otpauth://totp/... URI encoded in the QR")
    qr_code_data_uri: str = Field(..., description="data:image/png;base64,... QR image")
    backup_codes: list[str] = Field(..., description="One-time backup codes (shown once)")


class TwoFactorCodeBody(BaseModel):
    code: str = Field(..., min_length=6, max_length=10, description="6-digit TOTP or backup code")


class TwoFactorVerifyResponse(BaseModel):
    enabled: bool


class TwoFactorStatusResponse(BaseModel):
    enabled: bool
    enrolled_at: str | None


# ── Helpers ─────────────────────────────────────────────────────────────────

ISSUER = "Mero Dafa"


def _generate_backup_codes(count: int = 10) -> list[str]:
    """
    Generate `count` base32-looking one-time codes. Format: 10-char groups,
    e.g. ABCD-EFGH, to make them legible in a password manager.
    """
    out: list[str] = []
    for _ in range(count):
        raw = secrets_module.token_hex(4).upper()  # 8 hex chars
        out.append(f"{raw[:4]}-{raw[4:]}")
    return out


def _make_qr_data_uri(otpauth_url: str) -> str:
    """Render a PNG QR for the otpauth URL and return as a data: URI."""
    img = qrcode.make(otpauth_url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


async def _match_backup_code(code: str, hashed_list: list[str]) -> int | None:
    """Return the index of the matching backup code, else None."""
    candidate = code.upper().strip()
    for idx, hashed in enumerate(hashed_list):
        if verify_password(candidate, hashed):
            return idx
    return None


def verify_totp_code(secret: str, code: str) -> bool:
    """Valid-window check with a tiny ±1-step tolerance."""
    return pyotp.TOTP(secret).verify(code.strip(), valid_window=1)


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/setup", response_model=TwoFactorSetupResponse)
async def setup_two_factor(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Start a 2FA enrollment: generate a fresh secret + 10 backup codes, return
    the QR code as a data URI, and stage them on the user doc pending
    verification. Calling this endpoint again overwrites the pending secret
    (the user hasn't activated the previous one yet — safe).
    """
    user_id = current_user["sub"]
    user = await db.users.find_one(
        {"user_id": user_id},
        {"_id": 0, "email": 1, "totp_enabled": 1},
    )
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.get("totp_enabled"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="2FA is already enabled. Disable it first to re-enrol.",
        )

    secret = pyotp.random_base32()
    backup_codes = _generate_backup_codes()
    hashed_backups = [hash_password(c) for c in backup_codes]
    otpauth_url = pyotp.TOTP(secret).provisioning_uri(
        name=user["email"], issuer_name=ISSUER
    )
    qr_data_uri = _make_qr_data_uri(otpauth_url)

    await db.users.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "totp_secret_pending": secret,
                "totp_backup_codes_pending": hashed_backups,
            }
        },
    )
    logger.info(f"2FA setup initiated for user {user_id}")
    return TwoFactorSetupResponse(
        secret=secret,
        otpauth_url=otpauth_url,
        qr_code_data_uri=qr_data_uri,
        backup_codes=backup_codes,
    )


@router.post("/verify", response_model=TwoFactorVerifyResponse)
async def verify_two_factor(
    body: TwoFactorCodeBody,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Finalize 2FA enrollment. The user's authenticator should now be producing
    codes for the pending secret; this call promotes it to the active secret
    and stores the (already-hashed) backup codes.
    """
    from datetime import datetime, timezone

    user_id = current_user["sub"]
    user = await db.users.find_one(
        {"user_id": user_id},
        {
            "_id": 0,
            "totp_secret_pending": 1,
            "totp_backup_codes_pending": 1,
            "totp_enabled": 1,
        },
    )
    if not user or not user.get("totp_secret_pending"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending 2FA setup. Call /user/2fa/setup first.",
        )
    if user.get("totp_enabled"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="2FA is already enabled.",
        )

    if not verify_totp_code(user["totp_secret_pending"], body.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code. Try again.",
        )

    await db.users.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "totp_secret": user["totp_secret_pending"],
                "totp_backup_codes": user.get("totp_backup_codes_pending", []),
                "totp_enabled": True,
                "totp_enrolled_at": datetime.now(timezone.utc),
            },
            "$unset": {
                "totp_secret_pending": "",
                "totp_backup_codes_pending": "",
            },
        },
    )
    logger.info(f"2FA enabled for user {user_id}")
    return TwoFactorVerifyResponse(enabled=True)


@router.post("/disable")
async def disable_two_factor(
    body: TwoFactorCodeBody,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Disable 2FA. Requires a current TOTP code OR an unused backup code — we
    don't let the user turn it off with only their password, to match the
    protection we promised them at enrolment.
    """
    user_id = current_user["sub"]
    user = await db.users.find_one(
        {"user_id": user_id},
        {
            "_id": 0,
            "totp_secret": 1,
            "totp_enabled": 1,
            "totp_backup_codes": 1,
        },
    )
    if not user or not user.get("totp_enabled") or not user.get("totp_secret"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is not enabled on this account.",
        )

    code_ok = verify_totp_code(user["totp_secret"], body.code)
    if not code_ok:
        idx = await _match_backup_code(body.code, user.get("totp_backup_codes", []))
        if idx is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid code. Use your authenticator app or a backup code.",
            )

    await db.users.update_one(
        {"user_id": user_id},
        {
            "$set": {"totp_enabled": False},
            "$unset": {
                "totp_secret": "",
                "totp_backup_codes": "",
                "totp_enrolled_at": "",
                "totp_secret_pending": "",
                "totp_backup_codes_pending": "",
            },
        },
    )
    logger.info(f"2FA disabled for user {user_id}")
    return {"enabled": False}


@router.post("/regenerate-backup-codes")
async def regenerate_backup_codes(
    body: TwoFactorCodeBody,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Rotate the backup codes. Requires a current TOTP code (not a backup code —
    using one backup code to regenerate the rest defeats the lockout-escape
    purpose of the full set).
    """
    user_id = current_user["sub"]
    user = await db.users.find_one(
        {"user_id": user_id},
        {"_id": 0, "totp_secret": 1, "totp_enabled": 1},
    )
    if not user or not user.get("totp_enabled") or not user.get("totp_secret"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is not enabled on this account.",
        )
    if not verify_totp_code(user["totp_secret"], body.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid code. Enter the 6-digit code from your authenticator app.",
        )

    new_codes = _generate_backup_codes()
    hashed = [hash_password(c) for c in new_codes]
    await db.users.update_one(
        {"user_id": user_id},
        {"$set": {"totp_backup_codes": hashed}},
    )
    logger.info(f"2FA backup codes regenerated for user {user_id}")
    return {"backup_codes": new_codes}


@router.get("/status", response_model=TwoFactorStatusResponse)
async def two_factor_status(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return a minimal view of whether 2FA is active on this account."""
    user_id = current_user["sub"]
    user = await db.users.find_one(
        {"user_id": user_id},
        {"_id": 0, "totp_enabled": 1, "totp_enrolled_at": 1},
    )
    enrolled_at = user.get("totp_enrolled_at") if user else None
    return TwoFactorStatusResponse(
        enabled=bool(user and user.get("totp_enabled")),
        enrolled_at=enrolled_at.isoformat() if enrolled_at else None,
    )
