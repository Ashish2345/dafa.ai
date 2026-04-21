"""
User Pydantic schemas for auth API request/response models.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Request body for user registration."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")
    full_name: str = Field(..., min_length=1, max_length=100, description="Full name")


class UserLogin(BaseModel):
    """Request body for user login."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="Password")


class UserResponse(BaseModel):
    """User profile returned in API responses (no sensitive fields)."""

    user_id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User email address")
    full_name: str = Field(..., description="Full name")
    role: str = Field(..., description="User role (user, admin)")
    is_active: bool = Field(..., description="Whether the account is active")
    is_verified: bool = Field(default=False, description="Whether email is verified")
    created_at: datetime = Field(..., description="Account creation timestamp")
    profile_photo_url: Optional[str] = Field(
        default=None,
        description="Relative URL to the user's uploaded profile photo (null if none)",
    )


class TokenResponse(BaseModel):
    """Response returned on successful login or token refresh."""

    access_token: str = Field(..., description="JWT access token (expires in 30 min by default)")
    refresh_token: str = Field(..., description="JWT refresh token (expires in 7 days by default)")
    token_type: str = Field(default="bearer", description="Token type")


class RefreshRequest(BaseModel):
    """Request body for token refresh."""

    refresh_token: str = Field(..., description="Valid refresh token")


class VerifyEmailRequest(BaseModel):
    """Request body for email verification."""

    email: EmailStr = Field(..., description="Email to verify")
    code: str = Field(..., min_length=6, max_length=6, description="6-digit verification code")


class ResendVerificationRequest(BaseModel):
    """Request body to resend verification code."""

    email: EmailStr = Field(..., description="Email to resend code to")


class ForgotPasswordRequest(BaseModel):
    """Request body to request password reset."""

    email: EmailStr = Field(..., description="Email address of the account")


class ResetPasswordRequest(BaseModel):
    """Request body to reset password with code."""

    email: EmailStr = Field(..., description="Email address of the account")
    code: str = Field(..., min_length=6, max_length=6, description="6-digit reset code")
    new_password: str = Field(..., min_length=8, description="New password (minimum 8 characters)")


class ChangePasswordRequest(BaseModel):
    """Request body for authenticated password change."""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (minimum 8 characters)")
