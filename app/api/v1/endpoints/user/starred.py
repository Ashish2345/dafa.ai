"""
Starred responses endpoints — per-user Q&A bookmarks.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.config.plan_loader import plan_catalog
from app.db.mongodb import get_database
from app.db.repositories.starred_repository import StarredRepository
from app.db.repositories.user_repository import UserRepository
from app.models.starred import StarredResponse, StarredResponseCreate, StarredResponseList
from app.utils.auth import get_current_user

router = APIRouter(prefix="/user/starred-responses", tags=["user"])


@router.get("", response_model=StarredResponseList)
async def list_starred(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return the current user's starred responses, newest first."""
    repo = StarredRepository(db)
    items = await repo.list_for_user(current_user["sub"])
    return StarredResponseList(items=items, total=len(items))


@router.post("", response_model=StarredResponse, status_code=status.HTTP_201_CREATED)
async def create_starred(
    body: StarredResponseCreate,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Star a response. Idempotent-ish: existing (workspace, query) match is returned.

    The retained count is capped by the user's plan `max_starred_responses`.
    When at the cap the oldest starred item is silently evicted.
    """
    repo = StarredRepository(db)
    existing = await repo.find_for_query(
        current_user["sub"], body.workspace_id, body.query
    )
    if existing:
        return StarredResponse(**existing)

    # Look up the user's plan cap
    user_doc = await UserRepository(db).get_by_id(current_user["sub"])
    plan_id = (user_doc or {}).get("plan") or plan_catalog.default_plan_id
    plan = plan_catalog.get(plan_id)

    created = await repo.create(
        current_user["sub"],
        body.model_dump(),
        max_count=plan.limits.max_starred_responses,
    )
    logger.info(f"User {current_user['sub']} starred a response in workspace {body.workspace_id}")
    return StarredResponse(**created)


@router.delete("/{starred_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_starred(
    starred_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Unstar a specific response. Returns 404 if not found (or not owned)."""
    repo = StarredRepository(db)
    removed = await repo.delete(current_user["sub"], starred_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Starred response not found",
        )
    return None


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def clear_all(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Clear all starred responses for the current user."""
    repo = StarredRepository(db)
    count = await repo.delete_all_for_user(current_user["sub"])
    logger.info(f"User {current_user['sub']} cleared {count} starred responses")
    return None
