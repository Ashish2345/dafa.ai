"""
Feedback / support schemas — user-submitted reports with optional screenshots.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


FeedbackCategory = Literal["bug", "wrong_answer", "feature", "ui", "other"]
FeedbackStatus = Literal["new", "triaged", "resolved", "wontfix"]


class FeedbackResponse(BaseModel):
    """A feedback item as returned after successful submission."""

    id: str
    category: FeedbackCategory
    message: str
    has_image: bool
    image_content_type: Optional[str] = None
    status: FeedbackStatus
    created_at: datetime


class FeedbackCreated(BaseModel):
    """Lightweight response after POST /feedback — just the ID + timestamp."""

    id: str
    status: FeedbackStatus = "new"
    created_at: datetime
    message: str = Field(default="Thanks — we got it.")
