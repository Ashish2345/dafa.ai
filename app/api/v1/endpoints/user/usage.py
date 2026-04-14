"""
Usage endpoint — returns the current user's daily usage + remaining quota.
"""

from fastapi import APIRouter, Depends

from app.config.plan_loader import plan_catalog
from app.db.mongodb import get_database
from app.db.repositories.usage_repository import UsageRepository
from app.db.repositories.user_repository import UserRepository
from app.models.usage import (
    DailyUsagePoint,
    UsageCount,
    UsageTodayResponse,
    UsageTrendResponse,
)
from app.utils.auth import get_current_user

router = APIRouter(prefix="/user", tags=["user"])


def _build_count(used: int, limit: int) -> UsageCount:
    if limit == -1:
        return UsageCount(used=used, limit=-1, remaining=-1, unlimited=True)
    remaining = max(0, limit - used)
    return UsageCount(used=used, limit=limit, remaining=remaining, unlimited=False)


@router.get("/usage/today", response_model=UsageTodayResponse)
async def get_today_usage(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Return what the current user has used today and how much is left,
    based on their plan's limits.
    """
    user_id = current_user["sub"]

    # Current plan for this user
    user_doc = await UserRepository(db).get_by_id(user_id)
    plan_id = (user_doc or {}).get("plan") or plan_catalog.default_plan_id
    plan = plan_catalog.get(plan_id)

    usage_repo = UsageRepository(db)
    counts = await usage_repo.get_today(user_id)

    return UsageTodayResponse(
        date=UsageRepository.today_utc(),
        plan_id=plan.id,
        searches=_build_count(counts.get("search", 0), plan.limits.searches_per_day),
        pdf_downloads=_build_count(
            counts.get("pdf_download", 0), plan.limits.pdf_downloads_per_day
        ),
    )


@router.get("/usage/last-7-days", response_model=UsageTrendResponse)
async def get_last_7_days_usage(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return per-day search & PDF download counts for the last 7 days (oldest first)."""
    usage_repo = UsageRepository(db)
    rows = await usage_repo.get_last_n_days(current_user["sub"], days=7)

    total_searches = sum(r["searches"] for r in rows)
    total_pdf_downloads = sum(r["pdf_downloads"] for r in rows)

    return UsageTrendResponse(
        days=[DailyUsagePoint(**r) for r in rows],
        total_searches=total_searches,
        total_pdf_downloads=total_pdf_downloads,
    )
