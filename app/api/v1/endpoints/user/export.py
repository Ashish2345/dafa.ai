"""
Data export endpoint — returns a single JSON document containing every piece
of user-scoped data the account has generated. Intended for the "Export my
data" button on the Preferences page. Binary document files are not included;
only metadata, because shipping the PDFs as base64 blows up the payload.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from app.db.mongodb import get_database
from app.db.repositories.chat_repository import ChatRepository
from app.db.repositories.preferences_repository import PreferencesRepository
from app.db.repositories.starred_repository import StarredRepository
from app.db.repositories.user_repository import UserRepository
from app.utils.auth import get_current_user

router = APIRouter(prefix="/user", tags=["user"])


def _json_safe(value: Any) -> Any:
    """Recursively convert Mongo-returned values to JSON-serializable types."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items() if k != "_id"}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@router.get("/export")
async def export_my_data(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Bundle everything the authenticated user owns into one JSON download.

    Includes: user profile, preferences, conversations + messages, starred
    responses, private document metadata, and feedback submissions. Does NOT
    include: binary document files (too large) or anything scoped to other
    users.
    """
    user_id = current_user["sub"]

    # 1. Profile (strip sensitive fields before exporting)
    user_doc = await UserRepository(db).get_by_id(user_id)
    if user_doc:
        user_doc.pop("profile_photo_base64", None)  # don't leak the raw blob
        user_doc = _json_safe(user_doc)

    # 2. Preferences
    prefs = await PreferencesRepository(db).get_for_user(user_id)
    prefs = _json_safe(prefs)

    # 3. Conversations + messages
    chat_repo = ChatRepository(db)
    conversations = await chat_repo.list_conversations(user_id, max_age_days=-1)
    messages_by_workspace: dict[str, list[dict]] = {}
    for conv in conversations:
        ws_id = conv.get("workspace_id")
        if not ws_id:
            continue
        msgs = await chat_repo.get_messages(user_id, ws_id)
        messages_by_workspace[ws_id] = _json_safe(msgs)
    conversations = _json_safe(conversations)

    # 4. Starred responses
    starred_cursor = db.starred_responses.find(
        {"user_id": user_id}, {"_id": 0, "user_id": 0}
    )
    starred = [_json_safe(d) async for d in starred_cursor]

    # 5. Private documents (metadata only — no binaries)
    docs_cursor = db.documents.find(
        {"user_id": user_id, "scope": "private"},
        {"_id": 0, "pages": 0, "images": 0},
    )
    documents = [_json_safe(d) async for d in docs_cursor]

    # 6. Feedback submissions (strip the screenshot bytes too)
    feedback_cursor = db.feedback.find(
        {"user_id": user_id},
        {"_id": 0, "image_base64": 0, "user_id": 0},
    )
    feedback = [_json_safe(d) async for d in feedback_cursor]

    now = datetime.now(timezone.utc)
    payload = {
        "export_version": 1,
        "generated_at": now.isoformat(),
        "user_id": user_id,
        "profile": user_doc,
        "preferences": prefs,
        "conversations": conversations,
        "messages_by_workspace": messages_by_workspace,
        "starred_responses": starred,
        "private_documents": documents,
        "feedback": feedback,
    }

    filename = f"merodafa-export-{user_id}-{now.strftime('%Y%m%d-%H%M%S')}.json"
    logger.info(
        f"Data export for user {user_id}: "
        f"{len(conversations)} convs, {len(starred)} starred, "
        f"{len(documents)} docs, {len(feedback)} feedback"
    )
    return JSONResponse(
        content=payload,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
