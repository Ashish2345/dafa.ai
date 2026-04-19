"""Per-plan quota check for query endpoints.

Extracted from inline endpoint code so both POST /query and POST /query/stream
enforce the same rule through one service.
"""

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, status

from app.config.plan_loader import plan_catalog
from app.db.repositories.usage_repository import UsageRepository
from app.db.repositories.user_repository import UserRepository


@dataclass(frozen=True)
class QuotaResult:
    """Outcome of a quota check for one user."""

    allowed: bool
    used: int
    remaining: int
    plan_id: str
    plan_name: str
    daily_limit: int


class QuotaService:
    """Reads plan + today's usage and decides whether a search is allowed."""

    def __init__(self, db: Any):
        self._user_repo = UserRepository(db)
        self._usage_repo = UsageRepository(db)

    async def check_search(self, user_id: str) -> QuotaResult:
        """Return the search quota state for ``user_id`` — does not increment."""
        user_doc = await self._user_repo.get_by_id(user_id)
        plan_id = (user_doc or {}).get("plan") or plan_catalog.default_plan_id
        plan = plan_catalog.get(plan_id)
        daily_limit = plan.limits.searches_per_day

        allowed, used, remaining = await self._usage_repo.check_quota(
            user_id, "search", daily_limit
        )
        return QuotaResult(
            allowed=allowed,
            used=used,
            remaining=remaining,
            plan_id=plan_id,
            plan_name=plan.name,
            daily_limit=daily_limit,
        )

    def raise_if_exceeded(self, result: QuotaResult) -> None:
        """Translate a denied quota result into the standard 402 response."""
        if result.allowed:
            return
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": "plan_limit_reached",
                "action": "search",
                "plan_id": result.plan_id,
                "plan_name": result.plan_name,
                "limit": result.daily_limit,
                "used": result.used,
                "message": (
                    f"You've used all {result.daily_limit} questions on your "
                    f"{result.plan_name} plan today. Upgrade to continue asking."
                ),
            },
        )

    async def increment_search(self, user_id: str) -> None:
        """Count one successful search toward the user's daily usage."""
        await self._usage_repo.increment(user_id, "search")

    async def increment_llm_usage(
        self, user_id: str, cost_usd: float, tokens: int
    ) -> None:
        """Accumulate aggregated LLM cost + tokens on today's usage counter."""
        if cost_usd > 0:
            await self._usage_repo.increment_float(user_id, "llm_cost_usd", cost_usd)
        if tokens > 0:
            await self._usage_repo.increment(user_id, "llm_tokens", amount=tokens)
