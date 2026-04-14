"""
Usage tracking schemas — daily counters per user.

A `PlanLimitReached` error is raised when a quota is exhausted so the frontend
can render an "upgrade" prompt.
"""

from typing import Literal

from pydantic import BaseModel, Field


Action = Literal["search", "pdf_download"]


class UsageCount(BaseModel):
    """A single action's count today + limit + remaining."""

    used: int = Field(..., description="How many times this action was used today")
    limit: int = Field(..., description="Plan limit for this action today (-1 unlimited)")
    remaining: int = Field(..., description="-1 if unlimited, else max(0, limit - used)")
    unlimited: bool = Field(..., description="True if limit == -1")


class UsageTodayResponse(BaseModel):
    """GET /user/usage/today — what the user has used today and how much is left."""

    date: str = Field(..., description="YYYY-MM-DD for which this usage applies")
    plan_id: str = Field(..., description="User's current plan id")
    searches: UsageCount
    pdf_downloads: UsageCount


class DailyUsagePoint(BaseModel):
    """One day's totals for the trend chart."""

    date: str = Field(..., description="YYYY-MM-DD")
    searches: int
    pdf_downloads: int


class UsageTrendResponse(BaseModel):
    """GET /user/usage/last-7-days — daily usage for the last N days."""

    days: list[DailyUsagePoint]
    total_searches: int
    total_pdf_downloads: int
