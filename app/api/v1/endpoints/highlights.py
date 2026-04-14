"""
Highlights endpoint — returns word-level bounding boxes for citation highlighting.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.db.mongodb import get_database
from app.db.repositories.ocr_bbox_repository import OcrBboxRepository
from app.utils.auth import get_current_user

router = APIRouter(prefix="/documents", tags=["highlights"])


def _find_node(nodes: list[dict], node_id: str) -> dict | None:
    """Recursively find a node by nodeId in the tree."""
    for node in nodes:
        if node.get("nodeId") == node_id:
            return node
        found = _find_node(node.get("children", []), node_id)
        if found:
            return found
    return None


def _group_words_by_line(words: list[dict]) -> list[dict]:
    """Group words by (block, line) and compute per-line bounding boxes."""
    line_map: dict[tuple[int, int], list[dict]] = {}
    for w in words:
        key = (w.get("block", 0), w.get("line", 0))
        line_map.setdefault(key, []).append(w)

    lines = []
    for (block, line), line_words in sorted(line_map.items()):
        bbox = {
            "x0": min(w["x0"] for w in line_words),
            "y0": min(w["y0"] for w in line_words),
            "x2": max(w["x2"] for w in line_words),
            "y2": max(w["y2"] for w in line_words),
        }
        lines.append({
            "line": line,
            "block": block,
            "bbox": bbox,
            "words": line_words,
        })
    return lines


@router.get("/{document_id}/highlights")
async def get_highlights(
    document_id: str,
    node_id: str = Query(..., description="Tree node ID to highlight"),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return word-level bounding boxes for a specific section."""
    tree_doc = await db.page_index_trees.find_one(
        {"document_id": document_id},
        {"tree.nodes": 1, "_id": 0},
    )
    if not tree_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    nodes = tree_doc.get("tree", {}).get("nodes", [])
    node = _find_node(nodes, node_id)
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Node '{node_id}' not found")

    start_char = node.get("start_char", -1)
    end_char = node.get("end_char", -1)
    if start_char < 0 or end_char < 0:
        return {"document_id": document_id, "node_id": node_id, "highlights": []}

    bbox_repo = OcrBboxRepository(db)
    page_results = await bbox_repo.get_words_in_range(document_id, start_char, end_char)

    highlights = []
    for page_data in page_results:
        lines = _group_words_by_line(page_data["words"])
        highlights.append({
            "page": page_data["page"],
            "lines": lines,
        })

    return {
        "document_id": document_id,
        "node_id": node_id,
        "highlights": highlights,
    }
