"""
Auth endpoints — register, login, token refresh, and current user profile.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.db.mongodb import get_database
from app.db.repositories.user_repository import UserRepository
from app.models.user import RefreshRequest, TokenResponse, UserCreate, UserLogin, UserResponse
from app.utils.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate, db=Depends(get_database)):
    """
    Create a new user account.

    Returns the user profile. Password is hashed with bcrypt and never returned.
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

    logger.info(f"New user registered: {body.email}")
    return UserResponse(**user)


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin, db=Depends(get_database)):
    """
    Authenticate with email and password.

    Returns JWT access token (30 min) and refresh token (7 days).
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

    access_token = create_access_token(
        user_id=user["user_id"],
        email=user["email"],
        role=user["role"],
    )
    refresh_token = create_refresh_token(user_id=user["user_id"])

    logger.info(f"User logged in: {body.email}")
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


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
