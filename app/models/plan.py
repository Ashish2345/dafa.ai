"""
Plan schemas — describes what each subscription tier includes.

Loaded at startup from config/plans.yaml by plan_loader.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PlanLimits(BaseModel):
    """Numeric caps for the plan. `-1` means unlimited."""

    searches_per_day: int = Field(..., description="Max questions per day (-1 unlimited)")
    pdf_downloads_per_day: int = Field(..., description="Max PDF downloads per day (-1 unlimited)")
    gazette_years: int = Field(..., description="Years of gazette archive (-1 full)")
    chat_history_days: int = Field(
        default=-1,
        description="How many days of chat history is kept (-1 = unlimited / no time filter)",
    )
    max_starred_responses: int = Field(
        default=50,
        description="Max starred responses retained (-1 unlimited)",
    )
    api_access: bool = Field(default=False, description="Can call the public API")


class Plan(BaseModel):
    """A subscription plan as returned by the API and loaded from YAML."""

    id: str = Field(..., description="Stable plan key (starter, pro, enterprise)")
    name: str
    description: str = ""
    price_npr: Optional[int] = Field(
        default=None,
        description="Monthly price in NPR. null = contact sales.",
    )
    period: Literal["monthly", "yearly", "custom"] = "monthly"
    highlight: bool = False
    order: int = 0
    limits: PlanLimits
    features: list[str] = Field(default_factory=list)


class PlanList(BaseModel):
    """GET /plans response."""

    plans: list[Plan]
    default_plan: str


class UserPlanResponse(BaseModel):
    """GET /user/plan response — current user's active plan."""

    plan: Plan
