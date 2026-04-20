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
    # Phase 14: act-level / doc-level scope hints. When ``scope_doc_id`` is set
    # the catalog is restricted to that document; when only ``scope_act_slug``
    # is set, the catalog is restricted to documents whose slug (title slug or
    # ``act_slug`` metadata) matches. Section-level scoping is deferred to
    # Phase 15 (requires a structural index change).
    scope_act_slug: str | None = Field(default=None)
    scope_doc_id: str | None = Field(default=None)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _build_catalog(
    db,
    *,
    current_user_id: str | None = None,
    scope_act_slug: str | None = None,
    scope_doc_id: str | None = None,
) -> list[CatalogEntry]:
    """
    Build the candidate catalog for Studio retrieval.

    Phase 14:
    - Visibility: public documents + the caller's private uploads.
    - ``scope_doc_id`` narrows to a single document.
    - ``scope_act_slug`` narrows to documents whose slug/title matches.
    """
    query: dict = {"status": "completed"}

    # Visibility: public or legacy + the caller's private uploads.
    if current_user_id is not None:
        query["$or"] = [
            {"scope": "public"},
            {"scope": {"$exists": False}},
            {"scope": "private", "user_id": current_user_id},
        ]
    else:
        query["$or"] = [{"scope": "public"}, {"scope": {"$exists": False}}]

    # Single-doc scope wins if supplied.
    if scope_doc_id:
        query["document_id"] = scope_doc_id

    cursor = db["documents"].find(query)
    entries: list[CatalogEntry] = []
    async for doc in cursor:
        category = doc.get("category")
        if category not in ("acts-rules", "finance-acts", "nrb", "ird", "gazette", "najirs"):
            continue

        doc_slug = (doc.get("act_slug") or doc.get("slug") or "") or None
        # Very light act-slug match: if the caller supplied ``scope_act_slug``
        # we keep only documents whose slug/title contains that slug (case-
        # insensitive substring). Structured slug index is Phase 15.
        if scope_act_slug:
            title = (doc.get("title") or "").lower()
            slug = (doc_slug or "").lower()
            needle = scope_act_slug.lower()
            if needle not in title and needle not in slug:
                continue

        doc_scope = doc.get("scope") or "public"
        entries.append(
            CatalogEntry(
                id=doc.get("document_id") or doc.get("title"),
                name=doc.get("title") or doc.get("document_id") or "",
                category=category,
                brief=doc.get("summary") or doc.get("description") or "",
                scope="private" if doc_scope == "private" else "public",
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

    catalog = await _build_catalog(
        db,
        current_user_id=current_user.get("sub"),
        scope_act_slug=request.scope_act_slug,
        scope_doc_id=request.scope_doc_id,
    )
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
