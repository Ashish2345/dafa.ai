"""Streaming query endpoint — delegates to ``QueryOrchestrator.stream_query``.

The full SSE event contract and business logic live in
:mod:`app.services.query.orchestrator`; this module only handles the HTTP
transport (auth, quota gate, ``StreamingResponse`` wiring).
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.db.mongodb import get_database
from app.models.schemas import StreamQueryRequest
from app.services.query.factory import build_query_orchestrator
from app.services.query.quota import QuotaService
from app.utils.auth import get_current_user

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/stream", summary="Ask a question (streaming progress)")
async def query_stream(
    request: StreamQueryRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Streaming query with real-time progress events via SSE."""
    user_id = current_user["sub"]
    quota = QuotaService(db)
    quota.raise_if_exceeded(await quota.check_search(user_id))

    orchestrator = build_query_orchestrator()

    return StreamingResponse(
        orchestrator.stream_query(req=request, user_id=user_id, quota=quota),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
