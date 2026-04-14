"""
Chat history endpoints — conversations sidebar + per-workspace messages.
"""

from fastapi import APIRouter, Depends, status
from loguru import logger

from app.config.plan_loader import plan_catalog
from app.db.mongodb import get_database
from app.db.repositories.chat_repository import ChatRepository
from app.db.repositories.user_repository import UserRepository
from app.models.chat import (
    ConversationItem,
    ConversationList,
    MessagesResponse,
    RecordTurnRequest,
    SaveMessagesRequest,
)
from app.utils.auth import get_current_user

router = APIRouter(prefix="/user/chats", tags=["user"])


@router.get("", response_model=ConversationList)
async def list_conversations(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """List the current user's chat conversations (sidebar), newest first.

    Respects the user's plan `chat_history_days` retention — on Starter only
    conversations touched within the last N days appear.
    """
    user_doc = await UserRepository(db).get_by_id(current_user["sub"])
    plan_id = (user_doc or {}).get("plan") or plan_catalog.default_plan_id
    plan = plan_catalog.get(plan_id)

    repo = ChatRepository(db)
    items = await repo.list_conversations(
        current_user["sub"],
        max_age_days=plan.limits.chat_history_days,
    )
    return ConversationList(items=items, total=len(items))


@router.post("/record-turn", response_model=ConversationItem)
async def record_turn(
    body: RecordTurnRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Called after a completed Q&A — upserts the conversation entry."""
    repo = ChatRepository(db)
    item = await repo.record_turn(
        user_id=current_user["sub"],
        workspace_id=body.workspace_id,
        workspace_label=body.workspace_label,
        query=body.query,
        answer_preview=body.answer_preview,
    )
    return ConversationItem(**item)


@router.get("/{workspace_id}/messages", response_model=MessagesResponse)
async def get_messages(
    workspace_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return the messages array for a workspace."""
    repo = ChatRepository(db)
    messages = await repo.get_messages(current_user["sub"], workspace_id)
    return MessagesResponse(workspace_id=workspace_id, messages=messages)


@router.put("/{workspace_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
async def save_messages(
    workspace_id: str,
    body: SaveMessagesRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Replace the entire messages array for a workspace."""
    repo = ChatRepository(db)
    await repo.save_messages(
        current_user["sub"],
        workspace_id,
        [m.model_dump(by_alias=True, exclude_none=True) for m in body.messages],
    )
    return None


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    workspace_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Remove a conversation and its messages."""
    repo = ChatRepository(db)
    await repo.delete_conversation(current_user["sub"], workspace_id)
    return None


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def clear_all(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Clear all chat history (conversations + messages) for the current user."""
    repo = ChatRepository(db)
    count = await repo.clear_all_for_user(current_user["sub"])
    logger.info(f"User {current_user['sub']} cleared {count} chat docs")
    return None
