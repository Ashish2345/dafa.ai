"""
Highlights endpoint — returns word-level bounding boxes for citation highlighting.

Returns data in the shape the frontend expects:
  { highlights: [{ node_id, title, page_bboxes: [{ page, bbox }] }], image_dimensions }

When word-level OCR data is available, each line gets its own page_bboxes entry
(tight per-line highlights). Falls back to section-level page_bboxes from the tree.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.db.mongodb import get_database
from app.db.repositories.ocr_bbox_repository import OcrBboxRepository
from app.utils.auth import get_current_user

router = APIRouter(prefix="/documents", tags=["highlights"])


def _find_node_by_int_id(nodes: list[dict], int_id: int) -> dict | None:
    """Recursively find a node by its depth-first integer ID."""
    for node in nodes:
        if node.get("id") == int_id:
            return node
        found = _find_node_by_int_id(node.get("children", []), int_id)
        if found:
            return found
    return None


def _group_words_into_line_bboxes(words: list[dict]) -> list[dict]:
    """Group words by (block, line) and return per-line bboxes.

    Returns list of {"page": int, "bbox": {"x0", "y0", "x2", "y2"}}
    with one entry per line (not per section).
    """
    line_map: dict[tuple[int, int], list[dict]] = {}
    for w in words:
        key = (w.get("block", 0), w.get("line", 0))
        line_map.setdefault(key, []).append(w)

    line_bboxes = []
    for (_block, _line), line_words in sorted(line_map.items()):
        line_bboxes.append({
            "x0": min(w["x0"] for w in line_words),
            "y0": min(w["y0"] for w in line_words),
            "x2": max(w["x2"] for w in line_words),
            "y2": max(w["y2"] for w in line_words),
        })
    return line_bboxes


@router.get("/{document_id}/highlights")
async def get_highlights(
    document_id: str,
    node_ids: int | None = Query(None, description="Integer node ID (depth-first)"),
    node_id: str | None = Query(None, description="String nodeId (e.g. '2.1')"),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return bounding boxes for citation highlighting.

    Response shape matches frontend HighlightsResponse:
    { highlights: [{ node_id, title, page_bboxes }], image_dimensions }
    """
    if node_ids is None and node_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either node_ids (int) or node_id (string)",
        )

    tree_doc = await db.page_index_trees.find_one(
        {"document_id": document_id},
        {"tree.nodes": 1, "image_dimensions": 1, "_id": 0},
    )
    if not tree_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    image_dimensions = tree_doc.get("image_dimensions")
    nodes = tree_doc.get("tree", {}).get("nodes", [])

    node = _find_node_by_int_id(nodes, node_ids) if node_ids is not None else None
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Node not found")

    node_int_id = node.get("id", node_ids)
    title = node.get("title", "")
    start_char = node.get("start_char", -1)
    end_char = node.get("end_char", -1)

    # Try word-level bboxes first (precise per-line highlights)
    # Word bboxes are normalized 0-1, but the frontend expects pixel coordinates
    # (it divides by image_dimensions to get 0-1). So multiply by dimensions.
    img_w = image_dimensions.get("width", 1) if image_dimensions else 1
    img_h = image_dimensions.get("height", 1) if image_dimensions else 1

    page_bboxes = []
    if start_char >= 0 and end_char >= 0:
        bbox_repo = OcrBboxRepository(db)
        page_results = await bbox_repo.get_words_in_range(document_id, start_char, end_char)

        for page_data in page_results:
            line_bboxes = _group_words_into_line_bboxes(page_data["words"])
            for bbox in line_bboxes:
                page_bboxes.append({
                    "page": page_data["page"],
                    "bbox": {
                        "x0": bbox["x0"] * img_w,
                        "y0": bbox["y0"] * img_h,
                        "x2": bbox["x2"] * img_w,
                        "y2": bbox["y2"] * img_h,
                    },
                })

    # Fallback: use section-level page_bboxes from tree node
    if not page_bboxes:
        page_bboxes = node.get("page_bboxes", [])

    return {
        "highlights": [
            {
                "node_id": node_int_id,
                "title": title,
                "page_bboxes": page_bboxes,
            }
        ],
        "image_dimensions": image_dimensions,
    }
