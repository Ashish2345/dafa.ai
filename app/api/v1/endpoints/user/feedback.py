"""
Feedback / support endpoint — accepts a message and optional screenshot.
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger

from app.db.mongodb import get_database
from app.db.repositories.feedback_repository import FeedbackRepository, MAX_IMAGE_BYTES
from app.db.repositories.user_repository import UserRepository
from app.models.feedback import FeedbackCreated
from app.utils.auth import get_current_user

router = APIRouter(prefix="/feedback", tags=["feedback"])

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
ALLOWED_CATEGORIES = {"bug", "wrong_answer", "feature", "ui", "other"}


@router.post("", response_model=FeedbackCreated, status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    message: str = Form(..., min_length=3, max_length=4000),
    category: str = Form(default="other"),
    image: Optional[UploadFile] = File(default=None),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Submit feedback, optionally with a screenshot.

    Multipart form fields:
    - `message` (required) — description of the issue (3-4000 chars)
    - `category` — one of: bug, wrong_answer, feature, ui, other (default: other)
    - `image` (optional) — PNG, JPEG, WebP, or GIF up to 3 MB
    """
    if category not in ALLOWED_CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid category. Must be one of: {', '.join(sorted(ALLOWED_CATEGORIES))}",
        )

    image_bytes: Optional[bytes] = None
    image_content_type: Optional[str] = None
    if image and image.filename:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported image type: {image.content_type}. Use PNG, JPEG, WebP, or GIF.",
            )
        image_bytes = await image.read()
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image too large. Maximum is {MAX_IMAGE_BYTES // (1024*1024)} MB.",
            )
        image_content_type = image.content_type

    # Resolve the user's email from the token's claims or the users collection
    email = current_user.get("email") or ""
    if not email:
        user = await UserRepository(db).get_by_id(current_user["sub"])
        email = (user or {}).get("email", "")

    repo = FeedbackRepository(db)
    try:
        created = await repo.create(
            user_id=current_user["sub"],
            email=email,
            category=category,
            message=message,
            image_bytes=image_bytes,
            image_content_type=image_content_type,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc

    logger.info(
        f"Feedback submitted by user {current_user['sub']}: {created['id']}"
    )
    return FeedbackCreated(
        id=created["id"],
        status=created["status"],
        created_at=created["created_at"],
    )
