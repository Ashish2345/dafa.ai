"""
Chat conversations & messages schemas.

Two resources:
- `chat_conversations` — sidebar entries (one per user × workspace)
- `chat_messages` — the messages array inside a conversation
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Conversations ──────────────────────────────────────────────────────


class ConversationItem(BaseModel):
    """Item in the chat-history sidebar."""

    id: str
    workspace_id: str
    workspace_label: str = ""
    title: str
    preview: str = ""
    last_activity_at: datetime


class ConversationList(BaseModel):
    items: list[ConversationItem]
    total: int


class RecordTurnRequest(BaseModel):
    """POST /user/chats/record-turn — record that a Q&A happened."""

    workspace_id: str = Field(..., min_length=1)
    workspace_label: str = Field(default="", max_length=200)
    query: str = Field(..., min_length=1, max_length=2000)
    answer_preview: str = Field(default="", max_length=1000)


# ── Messages ──────────────────────────────────────────────────────────


Role = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    """A single chat message."""

    id: str
    role: Role
    content: str
    timestamp: str
    sources: Optional[list[dict[str, Any]]] = None
    confidence_level: Optional[str] = Field(default=None, alias="confidenceLevel")
    warnings: Optional[list[str]] = None
    follow_ups: Optional[list[str]] = Field(default=None, alias="followUps")
    chunks: Optional[list[dict[str, Any]]] = None

    model_config = {"populate_by_name": True}


class MessagesResponse(BaseModel):
    """GET /user/chats/{workspace_id}/messages response."""

    workspace_id: str
    messages: list[ChatMessage]


class SaveMessagesRequest(BaseModel):
    """PUT /user/chats/{workspace_id}/messages — replace entire messages array."""

    messages: list[ChatMessage]
