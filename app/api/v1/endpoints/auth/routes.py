"""
Auth endpoints — register (with email verification), login, token refresh,
password reset, Google OAuth, and current user profile.
"""

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from app.db.mongodb import get_database
from app.db.repositories.user_repository import UserRepository
from app.db.repositories.verification_repository import VerificationRepository
from app.settings import settings as app_settings
from app.models.user import (
    ForgotPasswordRequest,
    RefreshRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
    VerifyEmailRequest,
)
from app.services.email import send_password_reset_code, send_verification_code
from app.utils.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Register ────────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate, db=Depends(get_database)):
    """
    Create a new user account and send a 6-digit verification code to their email.

    The user is created with `is_verified=False`. They must call `/auth/verify-email`
    with the code before they can log in.
    """
    repo = UserRepository(db)
    try:
        user = await repo.create_user(
            email=body.email,
            hashed_password=hash_password(body.password),
            full_name=body.full_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    # Generate and send verification code
    vr = VerificationRepository(db)
    code = await vr.create_code(body.email, purpose="email_verify")
    try:
        await send_verification_code(to=body.email, code=code, name=body.full_name)
    except Exception as e:
        logger.error(f"Failed to send verification email to {body.email}: {e}")
        # Don't block registration — they can resend

    logger.info(f"New user registered (unverified): {body.email}")
    return UserResponse(**user)


# ── Verify email ────────────────────────────────────────────────────────────

@router.post("/verify-email")
async def verify_email(body: VerifyEmailRequest, db=Depends(get_database)):
    """
    Verify email with the 6-digit code. Marks the user as verified so they can log in.
    """
    vr = VerificationRepository(db)
    valid = await vr.verify(body.email, body.code, purpose="email_verify")

    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code",
        )

    repo = UserRepository(db)
    await repo.set_verified(body.email)

    # Auto-login: return tokens so the user doesn't have to type password again
    user = await repo.get_by_email(body.email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    access_token = create_access_token(
        user_id=user["user_id"], email=user["email"], role=user["role"],
    )
    refresh_token = create_refresh_token(user_id=user["user_id"])

    logger.info(f"Email verified: {body.email}")
    return {"status": "verified", "access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}


# ── Resend verification code ───────────────────────────────────────────────

@router.post("/resend-verification")
async def resend_verification(body: ResendVerificationRequest, db=Depends(get_database)):
    """
    Resend a 6-digit verification code to the user's email.
    Only works for registered but unverified users.
    """
    repo = UserRepository(db)
    user = await repo.get_by_email(body.email)

    if not user:
        # Don't reveal whether the email exists
        return {"status": "sent"}

    if user.get("is_verified", False):
        return {"status": "already_verified"}

    vr = VerificationRepository(db)
    code = await vr.create_code(body.email, purpose="email_verify")
    try:
        await send_verification_code(to=body.email, code=code, name=user.get("full_name", ""))
    except Exception as e:
        logger.error(f"Failed to resend verification email to {body.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send email. Please try again.",
        ) from e

    return {"status": "sent"}


# ── Login ───────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin, db=Depends(get_database)):
    """
    Authenticate with email and password.

    Returns JWT access token (30 min) and refresh token (7 days).
    Requires email to be verified first.
    """
    repo = UserRepository(db)
    user = await repo.get_by_email(body.email)

    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    if not user.get("is_verified", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please check your inbox for the verification code.",
        )

    access_token = create_access_token(
        user_id=user["user_id"],
        email=user["email"],
        role=user["role"],
    )
    refresh_token = create_refresh_token(user_id=user["user_id"])

    logger.info(f"User logged in: {body.email}")
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


# ── Forgot password ─────────────────────────────────────────────────────────

@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest, db=Depends(get_database)):
    """
    Send a 6-digit password reset code to the user's email.
    Always returns success to avoid leaking whether an email is registered.
    """
    repo = UserRepository(db)
    user = await repo.get_by_email(body.email)

    if user:
        vr = VerificationRepository(db)
        code = await vr.create_code(body.email, purpose="password_reset")
        try:
            await send_password_reset_code(to=body.email, code=code)
        except Exception as e:
            logger.error(f"Failed to send password reset email to {body.email}: {e}")

    # Always return success — don't reveal if email exists
    return {"status": "sent"}


# ── Reset password ──────────────────────────────────────────────────────────

@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest, db=Depends(get_database)):
    """
    Verify the 6-digit code and set a new password.
    """
    vr = VerificationRepository(db)
    valid = await vr.verify(body.email, body.code, purpose="password_reset")

    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset code",
        )

    repo = UserRepository(db)
    updated = await repo.update_password(body.email, hash_password(body.new_password))

    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    logger.info(f"Password reset: {body.email}")
    return {"status": "password_updated"}


# ── Token refresh ───────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(body: RefreshRequest, db=Depends(get_database)):
    """
    Exchange a valid refresh token for a new access + refresh token pair.
    """
    payload = decode_token(body.refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type — provide a refresh token",
        )

    user_id: str = payload.get("sub", "")
    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    if not user.get("is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    access_token = create_access_token(
        user_id=user["user_id"],
        email=user["email"],
        role=user["role"],
    )
    new_refresh_token = create_refresh_token(user_id=user["user_id"])

    return TokenResponse(access_token=access_token, refresh_token=new_refresh_token)


# ── Current user ────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user), db=Depends(get_database)):
    """
    Return the profile of the currently authenticated user.
    """
    repo = UserRepository(db)
    user = await repo.get_by_id(current_user["sub"])

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserResponse(**user)


# ── Google OAuth ────────────────────────────────────────────────────────────

_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


@router.get("/google/login", summary="Get Google OAuth consent URL")
async def google_login(redirect_uri: str = Query(..., description="Frontend callback URL")):
    """
    Returns the Google OAuth consent URL.
    The frontend should redirect the user to this URL.
    """
    if not app_settings.google_client_id:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Google OAuth not configured")

    return {"url": str(httpx.URL("https://accounts.google.com/o/oauth2/v2/auth", params={
        "client_id": app_settings.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }))}


@router.post("/google/callback", summary="Exchange Google auth code for tokens")
async def google_callback(
    code: str = Query(..., description="Authorization code from Google"),
    redirect_uri: str = Query(..., description="Same redirect_uri used in /google/login"),
    db=Depends(get_database),
):
    """
    Exchange the Google authorization code for user info, then create or log in the user.
    Returns MeroDafa JWT tokens.
    """
    if not app_settings.google_client_id or not app_settings.google_client_secret:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Google OAuth not configured")

    # 1. Exchange code for Google access token
    async with httpx.AsyncClient() as http:
        token_res = await http.post(_GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": app_settings.google_client_id,
            "client_secret": app_settings.google_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        })

    if token_res.status_code != 200:
        logger.error(f"Google token exchange failed: {token_res.text}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google authentication failed")

    access_token = token_res.json().get("access_token")
    if not access_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No access token from Google")

    # 2. Get user info from Google
    async with httpx.AsyncClient() as http:
        userinfo_res = await http.get(_GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})

    if userinfo_res.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Failed to get Google user info")

    google_user = userinfo_res.json()
    email = google_user.get("email")
    name = google_user.get("name", "")

    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google account has no email")

    # 3. Find or create user
    repo = UserRepository(db)
    user = await repo.get_by_email(email)

    if not user:
        # New user — create with random password (they use Google to sign in)
        import secrets
        user = await repo.create_user(
            email=email,
            hashed_password=hash_password(secrets.token_urlsafe(32)),
            full_name=name,
        )
        # Google already verified their email
        await repo.set_verified(email)
        logger.info(f"New Google user created: {email}")
        user = await repo.get_by_email(email)
    elif not user.get("is_verified", False):
        await repo.set_verified(email)

    if not user.get("is_active", True):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    # 4. Issue MeroDafa tokens
    merodafa_access = create_access_token(user_id=user["user_id"], email=user["email"], role=user["role"])
    merodafa_refresh = create_refresh_token(user_id=user["user_id"])

    logger.info(f"Google login: {email}")
    return {"access_token": merodafa_access, "refresh_token": merodafa_refresh, "token_type": "bearer"}
