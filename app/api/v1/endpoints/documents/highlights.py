"""
Highlights endpoint — returns word-level bounding boxes for citation highlighting.

Returns data in the shape the frontend expects:
  { highlights: [{ node_id, title, page_bboxes: [{ page, bbox }] }], image_dimensions }

Strategy per node:
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


async def _get_node_highlights(
    node: dict,
    document_id: str,
    bbox_repo: OcrBboxRepository,
    img_w: float,
    img_h: float,
) -> dict:
    """Build highlight entry for a single node.

    Strategy priority:
      1. Spatial matching using page_bboxes (always accurate — uses real y-coords)
      2. Char offset range (backup — char offsets can drift for multi-word OCR blocks)
      3. Raw section-level page_bboxes (last resort)

    Spatial is primary because page_bboxes come from proportional page mapping
    and word y-coordinates from OCR are always correct. Char offsets can drift
    when OCR text blocks don't match the markdown word-by-word.
    """
    page_bboxes: list[dict] = []

    # Strategy 1 (primary): spatial matching using node's page_bboxes region
    node_page_bboxes = node.get("page_bboxes", [])
    for pb in node_page_bboxes:
        pg = pb["page"]
        bbox = pb["bbox"]
        norm_y0 = bbox["y0"] / img_h if img_h else 0
        norm_y2 = bbox["y2"] / img_h if img_h else 1

        # Skip bboxes covering >86% of the page — likely fallback artifacts
        if (norm_y2 - norm_y0) > 0.86:
            continue

        words = await bbox_repo.get_words_in_spatial_region(
            document_id, pg, norm_y0, norm_y2,
        )
        if words:
            line_bboxes = _group_words_into_line_bboxes(words)
            page_bboxes.extend(_to_pixel_bboxes(line_bboxes, pg, img_w, img_h))

    # Strategy 2 (backup): char offset range
    if not page_bboxes:
        start_char = node.get("start_char", -1)
        end_char = node.get("end_char", -1)
        if start_char >= 0 and end_char > start_char:
            page_results = await bbox_repo.get_words_in_range(document_id, start_char, end_char)
            for page_data in page_results:
                line_bboxes = _group_words_into_line_bboxes(page_data["words"])
                page_bboxes.extend(_to_pixel_bboxes(line_bboxes, page_data["page"], img_w, img_h))

    # Strategy 3 (last resort): raw section-level page_bboxes
    if not page_bboxes:
        page_bboxes = node_page_bboxes

    return {
        "node_id": node.get("id"),
        "title": node.get("title", ""),
        "page_bboxes": page_bboxes,
    }


@router.get("/{document_id}/highlights")
async def get_highlights(
    document_id: str,
    node_ids: str = Query(..., description="Comma-separated integer node IDs"),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return bounding boxes for citation highlighting."""
    try:
        ids = [int(x.strip()) for x in node_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="node_ids must be comma-separated integers")
    if not ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="node_ids required")

    tree_doc = await db.page_index_trees.find_one(
        {"document_id": document_id},
        {"tree.nodes": 1, "image_dimensions": 1, "_id": 0},
    )
    if not tree_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    image_dimensions = tree_doc.get("image_dimensions")
    nodes = tree_doc.get("tree", {}).get("nodes", [])
    img_w = image_dimensions.get("width", 1) if image_dimensions else 1
    img_h = image_dimensions.get("height", 1) if image_dimensions else 1
    bbox_repo = OcrBboxRepository(db)

    all_highlights = []
    for node_int_id in ids:
        node = _find_node_by_int_id(nodes, node_int_id)
        if not node:
            continue
        highlight = await _get_node_highlights(node, document_id, bbox_repo, img_w, img_h)
        all_highlights.append(highlight)

    return {
        "highlights": all_highlights,
        "image_dimensions": image_dimensions,
    }
