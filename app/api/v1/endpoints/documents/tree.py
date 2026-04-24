"""
Page-index tree endpoints — expose the stored PageIndex tree and per-node
markdown slices to the frontend drill view (Phase 17).

The backend already builds and persists hierarchical trees via
``PageIndexService`` / ``PageIndexRepository``. These endpoints simply surface
that data shape for the UI, stripping retrieval-only fields (start_text /
end_text / start_char / end_char) from the tree payload.

Endpoints:
    GET /documents/{document_id}/tree
        → whole tree, slimmed for navigation + PDF highlighting
    GET /documents/{document_id}/tree/{node_id}/markdown
        → markdown slice for a single node (by DFS integer id)
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.db.mongodb import get_database
from app.db.repositories.document_repository import DocumentRepository
from app.db.repositories.page_index_repository import PageIndexRepository
from app.utils.auth import get_current_user


router = APIRouter(prefix="/documents", tags=["tree"])


def _slim_node(node: dict) -> dict:
    """Return a frontend-safe copy of ``node`` with retrieval-only fields stripped.

    Kept:
        nodeId, id, title, summary, page_range, page_bboxes, children (recursed)

    Dropped:
        start_char, end_char, start_text, end_text
        (redundant with the per-node /markdown endpoint and bloat the payload)
    """
    slim: dict[str, Any] = {
        "nodeId": node.get("nodeId"),
        "id": node.get("id"),
        "title": node.get("title", ""),
    }
    if node.get("summary") is not None:
        slim["summary"] = node["summary"]
    if node.get("page_range") is not None:
        slim["page_range"] = node["page_range"]
    if node.get("page_bboxes") is not None:
        slim["page_bboxes"] = node["page_bboxes"]

    children = node.get("children") or []
    slim["children"] = [_slim_node(c) for c in children]
    return slim


def _caller_can_see(doc: dict, current_user: dict) -> bool:
    """Apply the Phase 14 visibility rules to a single document record.

    Public (or legacy records without a scope field) → visible to anyone.
    Private → visible only to the owner.
    """
    scope = doc.get("scope")
    if scope in (None, "public"):
        return True
    if scope == "private":
        return doc.get("user_id") == current_user.get("sub")
    # Unknown scope — default to restrictive.
    return False


def _find_node_by_int_id(nodes: list[dict], target: int) -> dict | None:
    """DFS walk returning the first node whose integer ``id`` == ``target``."""
    for node in nodes:
        if node.get("id") == target:
            return node
        found = _find_node_by_int_id(node.get("children") or [], target)
        if found is not None:
            return found
    return None


@router.get("/{document_id}/tree", summary="Fetch the page-index tree for a document")
async def get_document_tree(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return the stored page-index tree for a document.

    Visibility: public documents are visible to anyone; private documents are
    restricted to their owner (same rule as the documents listing).

    Response shape::

        {
          "document_id": "...",
          "document_title": "...",
          "language": "en" | "ne",
          "page_count": 87,
          "nodes": [ { nodeId, id, title, summary?, page_range,
                        page_bboxes?, children: [...] } ]
        }
    """
    doc_repo = DocumentRepository(db)
    doc = await doc_repo.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    if not _caller_can_see(doc, current_user):
        # Hide existence of private docs the caller doesn't own.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    pi_repo = PageIndexRepository(db)
    tree_doc = await pi_repo.get_tree(document_id)
    if not tree_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No page-index tree for document {document_id}",
        )

    inner = tree_doc.get("tree") or {}
    raw_nodes = inner.get("nodes") or []
    slim_nodes = [_slim_node(n) for n in raw_nodes]

    # Prefer the title stored on the documents record; fall back to the tree's
    # own title field or the filename.
    metadata = doc.get("metadata") or {}
    title = (
        doc.get("title")
        or metadata.get("title")
        or inner.get("title")
        or doc.get("filename")
        or document_id
    )
    page_count = metadata.get("page_count") or inner.get("page_count") or 0
    language = tree_doc.get("language") or inner.get("language") or metadata.get("language") or "en"

    return {
        "document_id": document_id,
        "document_title": title,
        "language": language,
        "page_count": page_count,
        "nodes": slim_nodes,
    }


@router.get(
    "/{document_id}/tree/{node_id}/markdown",
    summary="Fetch the markdown slice for a single tree node",
)
async def get_document_tree_node_markdown(
    document_id: str,
    node_id: int,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return the markdown text for a single DFS-indexed node.

    ``node_id`` is the integer ``id`` field on the tree node (not the dotted
    ``nodeId`` string). The backend slices the full document markdown using the
    node's stored ``start_char`` / ``end_char`` offsets.
    """
    doc_repo = DocumentRepository(db)
    doc = await doc_repo.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )
    if not _caller_can_see(doc, current_user):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    pi_repo = PageIndexRepository(db)
    tree_doc = await pi_repo.get_tree(document_id)
    if not tree_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No page-index tree for document {document_id}",
        )

    nodes = (tree_doc.get("tree") or {}).get("nodes") or []
    node = _find_node_by_int_id(nodes, node_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node {node_id} not found in tree for document {document_id}",
        )

    start_char = node.get("start_char")
    end_char = node.get("end_char")
    markdown_text = ""
    if (
        isinstance(start_char, int)
        and isinstance(end_char, int)
        and end_char > start_char
    ):
        full = await pi_repo.get_markdown(document_id)
        if full:
            # Clamp to bounds so a stale offset can't raise — an out-of-range
            # slice in Python just silently returns an empty string, but we
            # clamp explicitly so partial overlaps still return what we have.
            lo = max(0, start_char)
            hi = min(len(full), end_char)
            if hi > lo:
                markdown_text = full[lo:hi]
    else:
        logger.warning(
            f"Node {node_id} on {document_id} has missing/invalid char offsets "
            f"(start_char={start_char!r}, end_char={end_char!r}) — returning empty markdown"
        )

    return {
        "document_id": document_id,
        "node_id": node_id,
        "title": node.get("title", ""),
        "markdown": markdown_text,
    }
