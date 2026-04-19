"""Factory for assembling a fully-wired :class:`QueryOrchestrator`.

Today the orchestrator's dependencies all have sensible defaults, so this
factory is a one-liner. It exists so callers (endpoints, tests, background
jobs) have a single well-known name to reach for — and so we have a place
to add configuration (model overrides, per-tenant strategies, etc.) without
touching every call site.
"""

from app.services.query.followups import FollowupService
from app.services.query.orchestrator import QueryOrchestrator
from app.services.query.retrieval import RetrievalService
from app.services.query.synthesis import SynthesisService


def build_query_orchestrator() -> QueryOrchestrator:
    return QueryOrchestrator(
        retrieval=RetrievalService(),
        synthesis=SynthesisService(),
        followups=FollowupService(),
    )
