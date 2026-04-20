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

    # Phase 14 — richer profile fields persisted on the server so ProfilePage no
    # longer has to fake them in local state.
    display_name: str = Field(default="", description="Name shown in shared threads")
    phone: str = Field(default="", description="Contact phone number")
    license_number: str = Field(default="", description="ICAN / NBA / firm PAN license")
    practice_area: str = Field(default="", description="Free-text practice-area description")
    timezone: str = Field(default="", description="IANA timezone, e.g. Asia/Kathmandu")

    # Appearance
    theme: ThemeMode = Field(default="light", description="UI theme preference")
    language: Language = Field(default="en", description="UI language preference")
    font_size_px: int = Field(default=15, ge=12, le=18, description="Base font size (12-18px)")

    # Notification toggles
    notify_new_gazettes: bool = Field(default=True, description="Email when new gazettes are published")
    notify_product_updates: bool = Field(default=True, description="Email about product updates")
    notify_weekly_roundup: bool = Field(default=True, description="Weekly Rajpatra roundup email")
    notify_shared_thread_activity: bool = Field(
        default=True,
        description="Notify when teammates share or reply on a thread",
    )

    # Phase 14 — privacy toggles from the Preferences page.
    improve_retrieval: bool = Field(
        default=True,
        description="Opt in to using anonymised queries to improve the retriever",
    )
    product_analytics: bool = Field(
        default=True,
        description="Opt in to anonymous product-usage analytics",
    )

    # Phase 14 — thread history retention window. ``None`` means keep forever.
    thread_retention_days: Optional[int] = Field(
        default=None,
        description="Chat history retention in days (null = forever)",
    )

    # Pinned / starred workspaces (collection names)
    starred_collections: list[str] = Field(default_factory=list, description="Pinned collection names")

    # Metadata
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")


class PreferencesUpdate(BaseModel):
    """Partial update for PUT /user/preferences — all fields optional."""

    organization: Optional[str] = Field(default=None, max_length=200)
    role: Optional[str] = Field(default=None, max_length=100)

    display_name: Optional[str] = Field(default=None, max_length=120)
    phone: Optional[str] = Field(default=None, max_length=40)
    license_number: Optional[str] = Field(default=None, max_length=80)
    practice_area: Optional[str] = Field(default=None, max_length=120)
    timezone: Optional[str] = Field(default=None, max_length=60)

    theme: Optional[ThemeMode] = Field(default=None)
    language: Optional[Language] = Field(default=None)
    font_size_px: Optional[int] = Field(default=None, ge=12, le=18)

    notify_new_gazettes: Optional[bool] = Field(default=None)
    notify_product_updates: Optional[bool] = Field(default=None)
    notify_weekly_roundup: Optional[bool] = Field(default=None)
    notify_shared_thread_activity: Optional[bool] = Field(default=None)

    improve_retrieval: Optional[bool] = Field(default=None)
    product_analytics: Optional[bool] = Field(default=None)
    thread_retention_days: Optional[int] = Field(default=None, ge=0, le=36500)

    starred_collections: Optional[list[str]] = Field(default=None, max_length=100)
