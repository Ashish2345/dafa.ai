"""
Highlights endpoint — returns word-level bounding boxes for citation highlighting.

Returns data in the shape the frontend expects:
  { highlights: [{ node_id, title, page_bboxes: [{ page, bbox }] }], image_dimensions }

Strategy:
  1. If node has start_char/end_char → find words by character offset range
  2. Else if node has page_bboxes → find words by spatial overlap (y-range per page)
  3. Else → return the section-level page_bboxes as-is (last resort)
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
    """Group words by (block, line) and return per-line bboxes (normalized 0-1)."""
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


def _to_pixel_bboxes(line_bboxes: list[dict], page: int, img_w: float, img_h: float) -> list[dict]:
    """Convert normalized 0-1 bboxes to pixel coordinates for the frontend."""
    return [
        {
            "page": page,
            "bbox": {
                "x0": b["x0"] * img_w,
                "y0": b["y0"] * img_h,
                "x2": b["x2"] * img_w,
                "y2": b["y2"] * img_h,
            },
        }
        for b in line_bboxes
    ]


@router.get("/{document_id}/highlights")
async def get_highlights(
    document_id: str,
    node_ids: int | None = Query(None, description="Integer node ID (depth-first)"),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return bounding boxes for citation highlighting."""
    if node_ids is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="node_ids required")

    tree_doc = await db.page_index_trees.find_one(
        {"document_id": document_id},
        {"tree.nodes": 1, "image_dimensions": 1, "_id": 0},
    )
    if not tree_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    image_dimensions = tree_doc.get("image_dimensions")
    nodes = tree_doc.get("tree", {}).get("nodes", [])
    node = _find_node_by_int_id(nodes, node_ids)
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")

    img_w = image_dimensions.get("width", 1) if image_dimensions else 1
    img_h = image_dimensions.get("height", 1) if image_dimensions else 1
    bbox_repo = OcrBboxRepository(db)

    page_bboxes: list[dict] = []

    start_char = node.get("start_char", -1)
    end_char = node.get("end_char", -1)

    # Strategy 1: char offset range (most precise)
    if start_char >= 0 and end_char > start_char:
        page_results = await bbox_repo.get_words_in_range(document_id, start_char, end_char)
        for page_data in page_results:
            line_bboxes = _group_words_into_line_bboxes(page_data["words"])
            page_bboxes.extend(_to_pixel_bboxes(line_bboxes, page_data["page"], img_w, img_h))

    # Strategy 2: spatial overlap using node's page_bboxes region
    if not page_bboxes:
        node_page_bboxes = node.get("page_bboxes", [])
        for pb in node_page_bboxes:
            pg = pb["page"]
            bbox = pb["bbox"]
            # page_bboxes bbox values are in pixel coords — normalize to 0-1
            norm_y0 = bbox["y0"] / img_h if img_h else 0
            norm_y2 = bbox["y2"] / img_h if img_h else 1
            words = await bbox_repo.get_words_in_spatial_region(
                document_id, pg, norm_y0, norm_y2,
            )
            if words:
                line_bboxes = _group_words_into_line_bboxes(words)
                page_bboxes.extend(_to_pixel_bboxes(line_bboxes, pg, img_w, img_h))

    # Strategy 3: raw section-level page_bboxes (last resort)
    if not page_bboxes:
        page_bboxes = node.get("page_bboxes", [])

    return {
        "highlights": [
            {
                "node_id": node.get("id", node_ids),
                "title": node.get("title", ""),
                "page_bboxes": page_bboxes,
            }
        ],
        "image_dimensions": image_dimensions,
    }
