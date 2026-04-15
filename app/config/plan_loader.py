"""
Plan loader — reads config/plans.yaml once at startup and caches in memory.

After editing plans.yaml the backend must be restarted to pick up changes.
"""

from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from app.models.plan import Plan


# Project root → /dafa.ai/config/plans.yaml
_DEFAULT_PATH = Path(__file__).parent.parent.parent / "config" / "plans.yaml"


class PlanCatalog:
    """In-memory catalog of plans loaded from YAML."""

    def __init__(self) -> None:
        self._plans: dict[str, Plan] = {}
        self._default_plan_id: str = "starter"
        self._loaded: bool = False

    def load(self, path: Optional[Path] = None) -> None:
        """Read and validate plans.yaml. Safe to call multiple times."""
        target = path or _DEFAULT_PATH
        if not target.exists():
            raise FileNotFoundError(f"Plan config not found: {target}")

        with target.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        default_id = raw.get("default_plan") or "starter"
        plans_raw = raw.get("plans") or []

        parsed: dict[str, Plan] = {}
        for entry in plans_raw:
            plan = Plan(**entry)
            if plan.id in parsed:
                raise ValueError(f"Duplicate plan id in {target}: {plan.id}")
            parsed[plan.id] = plan

        if default_id not in parsed:
            raise ValueError(
                f"default_plan '{default_id}' not found among plans: "
                f"{list(parsed.keys())}"
            )

        self._plans = parsed
        self._default_plan_id = default_id
        self._loaded = True
        logger.info(
            f"Loaded {len(parsed)} plans from {target.name}: "
            f"{', '.join(parsed.keys())} (default: {default_id})"
        )

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def all_plans(self) -> list[Plan]:
        """Return all plans sorted by `order` field."""
        self._ensure_loaded()
        return sorted(self._plans.values(), key=lambda p: p.order)

    def get(self, plan_id: str) -> Plan:
        """
        Return a plan by ID. Falls back to the default plan if the ID is unknown
        (e.g., a user has a legacy plan name that was removed from YAML).
        """
        self._ensure_loaded()
        plan = self._plans.get(plan_id)
        if plan is None:
            logger.warning(
                f"Unknown plan id '{plan_id}', falling back to default "
                f"'{self._default_plan_id}'"
            )
            return self._plans[self._default_plan_id]
        return plan

    @property
    def default_plan_id(self) -> str:
        self._ensure_loaded()
        return self._default_plan_id


# Module-level singleton
plan_catalog = PlanCatalog()
