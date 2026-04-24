"""
Legal domain catalog — the high-level taxonomy that groups documents into
browsable areas on the Home surface (Tax & Revenue, Banking & NRB, etc.).

Separate from `domain.py` which holds document-entity models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class LegalDomain(BaseModel):
    """One row per browsable legal area.

    Consumed by `GET /domains` and referenced by `Document.domain_slug`.
    Icon names come from the Lucide catalog the frontend knows how to resolve.
    """

    slug: str = Field(..., description="Unique kebab-case identifier, e.g. 'tax'")
    name: str = Field(..., description="Display name, e.g. 'Tax & Revenue'")
    icon: str = Field(..., description="Lucide icon name, e.g. 'Banknote'")
    icon_color_bg: str = Field(..., description="Hex background for the icon tile")
    icon_color_fg: str = Field(..., description="Hex foreground (icon stroke) colour")
    description: Optional[str] = None
    display_order: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class LegalDomainPatch(BaseModel):
    """Partial update payload for admin/operator edits."""

    name: Optional[str] = None
    icon: Optional[str] = None
    icon_color_bg: Optional[str] = None
    icon_color_fg: Optional[str] = None
    description: Optional[str] = None
    display_order: Optional[int] = None
