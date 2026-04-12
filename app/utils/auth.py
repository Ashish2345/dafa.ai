"""
JWT and password utilities for user authentication.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt as _bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from loguru import logger

from app.settings import settings

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def _to_bytes(plain: str) -> bytes:
    """Encode and truncate to 72 bytes — bcrypt silently ignores the rest."""
    return plain.encode("utf-8")[:72]


def hash_password(plain: str) -> str:
    """Hash a plain-text password using bcrypt."""
    return _bcrypt.hashpw(_to_bytes(plain), _bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the bcrypt hash."""
    return _bcrypt.checkpw(_to_bytes(plain), hashed.encode("utf-8"))


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def _create_token(data: dict, expires_delta: timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + expires_delta
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str, email: str, role: str) -> str:
    return _create_token(
        {"sub": user_id, "email": email, "role": role, "type": "access"},
        timedelta(minutes=settings.access_token_expire_minutes),
    )


def create_refresh_token(user_id: str) -> str:
    return _create_token(
        {"sub": user_id, "type": "refresh"},
        timedelta(days=settings.refresh_token_expire_days),
    )


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT.

    Raises:
        HTTPException 401 on invalid/expired token.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ---------------------------------------------------------------------------
# FastAPI dependency — current user
# ---------------------------------------------------------------------------

_DEV_USER = {
    "sub": "00000000-0000-0000-0000-000000000001",
    "email": "dev@local",
    "role": "admin",
    "type": "access",
}


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> dict:
    """
    FastAPI dependency that validates the Bearer token and returns the user payload.

    The returned dict contains: sub (user_id), email, role, type.

    In development mode (ENVIRONMENT=development) the token is optional —
    requests without a token are treated as the built-in dev user so you can
    call protected endpoints without registering/logging in.
    """
    if token is None:
        if settings.is_development:
            logger.debug("Dev bypass: no token supplied, returning dev user")
            return _DEV_USER
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type — use an access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload
