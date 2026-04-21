"""
Profile photo endpoints — upload, fetch, and clear the authenticated user's
profile picture.

Storage: images are stored as base64 on the user document itself, matching the
pattern used by the feedback screenshot uploader (see
`app/db/repositories/feedback_repository.py`). The 2 MB cap keeps us well under
Mongo's 16 MB doc limit and avoids adding a GridFS dependency. If usage grows
we can migrate to disk / S3 later — callers already hit the photo through an
opaque URL so the migration surface is small.
"""

import base64
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from loguru import logger

from app.db.mongodb import get_database
from app.utils.auth import get_current_user

router = APIRouter(prefix="/user/profile", tags=["user"])

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_PHOTO_BYTES = 2 * 1024 * 1024  # 2 MB


@router.post("/photo", status_code=status.HTTP_200_OK)
async def upload_profile_photo(
    file: UploadFile = File(..., description="Image file, max 2 MB"),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Upload a new profile photo for the authenticated user. PNG, JPEG, WebP, or
    GIF up to 2 MB. Replaces any existing photo.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported image type: {file.content_type}. "
                "Use PNG, JPEG, WebP, or GIF."
            ),
        )

    blob = await file.read()
    if len(blob) > MAX_PHOTO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large. Maximum is {MAX_PHOTO_BYTES // (1024 * 1024)} MB.",
        )
    if len(blob) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file.",
        )

    user_id = current_user["sub"]
    encoded = base64.b64encode(blob).decode("ascii")
    url = f"/user/profile/photo?v={len(blob)}"  # cheap cache-buster

    await db.users.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "profile_photo_base64": encoded,
                "profile_photo_content_type": file.content_type,
                "profile_photo_size_bytes": len(blob),
                "profile_photo_url": url,
            },
        },
    )
    logger.info(
        f"Profile photo uploaded for user {user_id} "
        f"({file.content_type}, {len(blob)} bytes)"
    )
    return {"url": url, "size_bytes": len(blob), "content_type": file.content_type}


@router.get("/photo")
async def get_profile_photo(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Stream the authenticated user's profile photo."""
    user_id = current_user["sub"]
    doc = await db.users.find_one(
        {"user_id": user_id},
        {
            "_id": 0,
            "profile_photo_base64": 1,
            "profile_photo_content_type": 1,
        },
    )
    if not doc or not doc.get("profile_photo_base64"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No profile photo set.",
        )
    blob = base64.b64decode(doc["profile_photo_base64"])
    content_type = doc.get("profile_photo_content_type") or "image/png"
    return Response(
        content=blob,
        media_type=content_type,
        headers={"Cache-Control": "private, max-age=60"},
    )


@router.delete("/photo", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile_photo(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Remove the authenticated user's profile photo."""
    user_id = current_user["sub"]
    await db.users.update_one(
        {"user_id": user_id},
        {
            "$unset": {
                "profile_photo_base64": "",
                "profile_photo_content_type": "",
                "profile_photo_size_bytes": "",
                "profile_photo_url": "",
            },
        },
    )
    logger.info(f"Profile photo removed for user {user_id}")
    return None
