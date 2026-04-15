"""
Plans endpoints — list available plans and fetch the current user's plan.
"""

from fastapi import APIRouter, Depends

from app.config.plan_loader import plan_catalog
from app.db.mongodb import get_database
from app.db.repositories.user_repository import UserRepository
from app.models.plan import PlanList, UserPlanResponse
from app.utils.auth import get_current_user

router = APIRouter(tags=["plans"])


@router.get("/plans", response_model=PlanList)
async def list_plans():
    """
    Return all available plans (public — no auth required).
    Used by the billing page and marketing pages to show pricing.
    """
    return PlanList(
        plans=plan_catalog.all_plans(),
        default_plan=plan_catalog.default_plan_id,
    )


@router.get("/user/plan", response_model=UserPlanResponse)
async def get_current_user_plan(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """
    Return the plan currently assigned to the authenticated user.
    Plan ID is read from `users.plan` field; falls back to default if unset.
    """
    repo = UserRepository(db)
    user = await repo.get_by_id(current_user["sub"])
    plan_id = (user or {}).get("plan") or plan_catalog.default_plan_id
    plan = plan_catalog.get(plan_id)
    return UserPlanResponse(plan=plan)
