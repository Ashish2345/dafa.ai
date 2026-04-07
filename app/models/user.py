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
    created_at: datetime = Field(..., description="Account creation timestamp")


class TokenResponse(BaseModel):
    """Response returned on successful login or token refresh."""

    access_token: str = Field(..., description="JWT access token (expires in 30 min by default)")
    refresh_token: str = Field(..., description="JWT refresh token (expires in 7 days by default)")
    token_type: str = Field(default="bearer", description="Token type")


class RefreshRequest(BaseModel):
    """Request body for token refresh."""

    refresh_token: str = Field(..., description="Valid refresh token")
