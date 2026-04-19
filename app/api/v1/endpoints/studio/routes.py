"""Studio query endpoint — SSE stream wrapping the StudioRouter."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.db.mongodb import get_database
from app.services.llm import LLMService
from app.services.studio.aggregator import DocumentAggregator
from app.services.studio.classifier import CatalogEntry, SourceType, TypeClassifier
from app.services.studio.router import StudioRouter
from app.utils.auth import get_current_user


router = APIRouter(prefix="/studio", tags=["studio"])


class _Turn(BaseModel):
    role: str
    content: str


class StudioQueryRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    history: list[_Turn] = Field(default_factory=list)
    pinned_types: list[SourceType] = Field(default_factory=list)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _build_catalog(db) -> list[CatalogEntry]:
    cursor = db["documents"].find({"status": "completed"})
    entries: list[CatalogEntry] = []
    async for doc in cursor:
        category = doc.get("category")
        if category not in ("acts-rules", "finance-acts", "nrb", "ird", "gazette", "najirs"):
            continue
        entries.append(
            CatalogEntry(
                id=doc.get("document_id") or doc.get("title"),
                name=doc.get("title") or doc.get("document_id") or "",
                category=category,
                brief=doc.get("summary") or doc.get("description") or "",
            )
        )
    return entries


@router.post("/query", summary="Studio — unified cross-source query")
async def studio_query(
    request: StudioQueryRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    llm = LLMService()
    aggregator = DocumentAggregator()
    classifier = TypeClassifier(llm=llm)
    router_svc = StudioRouter(classifier=classifier, aggregator=aggregator, llm=llm)

    catalog = await _build_catalog(db)
    history = [(t.role, t.content) for t in request.history]

    async def stream():
        try:
            async for evt in router_svc.run(
                query=request.message,
                catalog=catalog,
                history=history,
                pinned_types=request.pinned_types,
            ):
                yield _sse(evt.kind, evt.payload)
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
