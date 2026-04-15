"""
User preferences Pydantic schemas — theme, notifications, profile extras.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


ThemeMode = Literal["light", "dark", "system"]
Language = Literal["en", "ne"]


class PreferencesResponse(BaseModel):
    """User preferences returned by GET /user/preferences."""

    # Profile extras
    organization: str = Field(default="", description="Firm / organization name")
    role: str = Field(default="", description="Professional role, e.g. Chartered Accountant")

    # Appearance
    theme: ThemeMode = Field(default="light", description="UI theme preference")
    language: Language = Field(default="en", description="UI language preference")
    font_size_px: int = Field(default=15, ge=12, le=18, description="Base font size (12-18px)")

    # Notification toggles
    notify_new_gazettes: bool = Field(default=True, description="Email when new gazettes are published")
    notify_product_updates: bool = Field(default=True, description="Email about product updates")
    notify_weekly_roundup: bool = Field(default=True, description="Weekly Rajpatra roundup email")

    # Pinned / starred workspaces (collection names)
    starred_collections: list[str] = Field(default_factory=list, description="Pinned collection names")

    # Metadata
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")


class PreferencesUpdate(BaseModel):
    """Partial update for PUT /user/preferences — all fields optional."""

    organization: Optional[str] = Field(default=None, max_length=200)
    role: Optional[str] = Field(default=None, max_length=100)
    theme: Optional[ThemeMode] = Field(default=None)
    language: Optional[Language] = Field(default=None)
    font_size_px: Optional[int] = Field(default=None, ge=12, le=18)
    notify_new_gazettes: Optional[bool] = Field(default=None)
    notify_product_updates: Optional[bool] = Field(default=None)
    notify_weekly_roundup: Optional[bool] = Field(default=None)
    starred_collections: Optional[list[str]] = Field(default=None, max_length=100)
