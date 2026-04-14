"""
User preferences endpoints — GET and PUT for per-user settings.
"""

from fastapi import APIRouter, Depends
from loguru import logger

from app.db.mongodb import get_database
from app.db.repositories.preferences_repository import PreferencesRepository
from app.models.preferences import PreferencesResponse, PreferencesUpdate
from app.utils.auth import get_current_user

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/preferences", response_model=PreferencesResponse)
async def get_preferences(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Return the current user's preferences.
    Falls back to defaults if the user has never saved preferences.
    """
    repo = PreferencesRepository(db)
    prefs = await repo.get_for_user(current_user["sub"])
    return PreferencesResponse(**prefs)


@router.put("/preferences", response_model=PreferencesResponse)
async def update_preferences(
    body: PreferencesUpdate,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Update (upsert) the current user's preferences.
    Only the fields included in the request body are changed.
    """
    repo = PreferencesRepository(db)
    prefs = await repo.update_for_user(
        current_user["sub"],
        body.model_dump(exclude_unset=True),
    )
    logger.info(f"Preferences updated for user {current_user['sub']}")
    return PreferencesResponse(**prefs)
