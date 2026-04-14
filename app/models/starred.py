"""
Starred responses — schemas for saving & listing user-pinned Q&A answers.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class StarredResponseCreate(BaseModel):
    """Request body for POST /user/starred-responses."""

    workspace_id: str = Field(..., description="Workspace / collection ID where the query was asked")
    workspace_label: str = Field(default="", max_length=200, description="Human-readable workspace name")
    query: str = Field(..., min_length=1, max_length=2000, description="The question text")
    answer: str = Field(..., min_length=1, description="The answer text (may include markdown/citations)")
    sources: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional list of source citation objects from the answer",
    )


class StarredResponse(BaseModel):
    """A starred response as returned by GET /user/starred-responses."""

    id: str = Field(..., description="Unique starred-response ID")
    workspace_id: str
    workspace_label: str = ""
    query: str
    answer: str
    sources: Optional[list[dict[str, Any]]] = None
    starred_at: datetime = Field(..., description="When the user starred this response")


class StarredResponseList(BaseModel):
    """Response envelope for GET /user/starred-responses."""

    items: list[StarredResponse]
    total: int
