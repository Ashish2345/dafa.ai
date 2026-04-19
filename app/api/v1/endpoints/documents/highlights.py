"""
Highlights endpoint — returns word-level bounding boxes for citation highlighting.

Returns data in the shape the frontend expects:
  { highlights: [{ node_id, title, page_bboxes: [{ page, bbox }] }], image_dimensions }

Strategy per node (first hit wins):
  0. start_text / end_text phrase match directly on OCR words (cross-page aware)
  1. start_char / end_char → words whose char_offset falls in range
  2. Title y-search on first page, bounded by next section-number marker
  3. Raw section-level page_bboxes from the tree (last resort — one big rect)
"""

import re
from difflib import SequenceMatcher

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

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


def _dfs_flatten(nodes: list[dict]) -> list[dict]:
    """Flatten a tree into depth-first order."""
    result: list[dict] = []
    for node in nodes:
        result.append(node)
        result.extend(_dfs_flatten(node.get("children", [])))
    return result


def _find_next_title(nodes: list[dict], int_id: int) -> str | None:
    """Return the title of the node that follows *int_id* in DFS order.

    Used to determine where the current section ends on a shared page.
    """
    flat = _dfs_flatten(nodes)
    for i, node in enumerate(flat):
        if node.get("id") == int_id and i + 1 < len(flat):
            return flat[i + 1].get("title", "")
    return None



def _find_title_y_position(title: str, words: list[dict]) -> float | None:
    """Find the normalized y-coordinate where a section title begins on a page.

    Tries each title token as an anchor (OCR text may omit words like "दफा")
    and verifies by checking if another token appears nearby.
    """
    if not words or not title:
        return None

    tokens = title.replace(":", " ").replace("(", " ").replace(")", " ").split()
    tokens = [t.strip() for t in tokens if t.strip()]
    if not tokens:
        return None

    sorted_words = sorted(words, key=lambda w: (w.get("y0", 0), w.get("x0", 0)))

    # Try each token as the anchor — OCR may not have the first token at all
    for anchor_idx, anchor_token in enumerate(tokens[:4]):
        other_tokens = [t for j, t in enumerate(tokens[:4]) if j != anchor_idx]

        for i, w in enumerate(sorted_words):
            if anchor_token not in w.get("text", ""):
                continue

            # Single-token title — anchor match is enough
            if not other_tokens:
                return w["y0"]

            # Verify: at least one other token in nearby words (within 5 positions)
            for j in range(max(0, i - 3), min(len(sorted_words), i + 6)):
                if j == i:
                    continue
                for ot in other_tokens:
                    if ot in sorted_words[j].get("text", ""):
                        return w["y0"]

    return None


def _find_next_section_marker_y(words: list[dict], title_y: float) -> float | None:
    """Find the y-position of the next numbered section after title_y.

    Looks for OCR blocks starting with a number pattern (e.g. "८८.", "89.")
    which indicate the start of a new section on the same page.
    """
    sorted_words = sorted(words, key=lambda w: (w.get("y0", 0), w.get("x0", 0)))

    for w in sorted_words:
        wy = w.get("y0", 0)
        if wy <= title_y + 0.02:  # skip current section and above
            continue
        text = w.get("text", "").strip()
        # Match: Devanagari digits (०-९) or ASCII digits followed by "." or ":"
        if re.match(r'^[०-९\d]+[\.:]', text):
            return wy

    return None


async def _find_section_by_title(
    node: dict,
    document_id: str,
    bbox_repo: "OcrBboxRepository",
    img_w: float,
    img_h: float,
    next_title: str | None = None,
) -> list[dict]:
    """Locate a section by searching for its title in OCR words on the target page.

    Highlights from the title's y-position down to either:
      - the next section's title (if found on the same page), or
      - the end of the page.
    """
    title = node.get("title", "").strip()
    page_range = node.get("page_range", [])
    if not title or not page_range:
        return []

    start_page = page_range[0]
    end_page = page_range[-1] if len(page_range) >= 2 else start_page
    # Cap to avoid over-highlighting broad chapter nodes
    end_page = min(end_page, start_page + 2)

    words = await bbox_repo.get_page_words(document_id, start_page)
    if not words:
        return []

    title_y = _find_title_y_position(title, words)
    if title_y is None:
        return []

    # Find where the NEXT section starts on the same page (end boundary).
    # Use section-number pattern matching (e.g. "८८.") — more reliable than
    # tree title matching since OCR text rarely matches human-written titles.
    end_y: float = 1.1  # default: end of page
    next_y = _find_next_section_marker_y(words, title_y)
    if next_y is not None:
        end_y = next_y

    result: list[dict] = []

    # First page: highlight from title_y to end_y.
    # Convert each OCR block directly to a pixel bbox (don't use
    # _group_words_into_line_bboxes — OCR data is block-level, not
    # word-level, so line grouping produces fragmented results).
    section_words = [
        w for w in words
        if w.get("y0", 0) >= title_y - 0.03 and w.get("y0", 0) < end_y
    ]
    for w in section_words:
        result.append({
            "page": start_page,
            "bbox": {
                "x0": w["x0"] * img_w,
                "y0": w["y0"] * img_h,
                "x2": w["x2"] * img_w,
                "y2": w["y2"] * img_h,
            },
        })

    # Remaining pages in range (only if end_y was not found on first page,
    # meaning the section continues beyond the first page)
    if end_y > 1.0:
        for pg in range(start_page + 1, end_page + 1):
            pg_words = await bbox_repo.get_page_words(document_id, pg)
            if not pg_words:
                continue
            # On the last page, check if the next section starts here
            pg_end_y: float = 1.1
            if next_title:
                ny = _find_title_y_position(next_title, pg_words)
                if ny is not None:
                    pg_end_y = ny
            for w in pg_words:
                if w.get("y0", 0) < pg_end_y:
                    result.append({
                        "page": pg,
                        "bbox": {
                            "x0": w["x0"] * img_w,
                            "y0": w["y0"] * img_h,
                            "x2": w["x2"] * img_w,
                            "y2": w["y2"] * img_h,
                        },
                    })

    return result


_ANCHOR_PUNCT_RE = re.compile(r'[।॥:;,\.\(\)\[\]\-—–\'"`]')


def _normalize_for_match(text: str) -> str:
    """Length-preserving normalization so char offsets stay stable."""
    return _ANCHOR_PUNCT_RE.sub(' ', text).lower()


def _sanitize_anchor(raw: str) -> str:
    """Strip markdown artifacts and collapse whitespace in a tree anchor."""
    text = re.sub(r'<!--.*?-->', '', raw)
    return re.sub(r'\s+', ' ', text).strip()


def _build_page_text(words: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate OCR words in reading order.

    Returns the joined string plus a list of ``(char_start, char_end, word_idx)``
    tuples letting a char offset be resolved back to the originating word.
    """
    ordered = sorted(
        enumerate(words),
        key=lambda p: (p[1].get("line", 0), p[1].get("y0", 0.0), p[1].get("x0", 0.0)),
    )
    parts: list[str] = []
    positions: list[tuple[int, int, int]] = []
    cursor = 0
    for orig_idx, w in ordered:
        text = str(w.get("text", "") or "").strip()
        if not text:
            continue
        if parts:
            parts.append(" ")
            cursor += 1
        parts.append(text)
        positions.append((cursor, cursor + len(text), orig_idx))
        cursor += len(text)
    return "".join(parts), positions


def _find_anchor_span(
    anchor: str, page_text: str, after_char: int = 0,
) -> tuple[int, int] | None:
    """Locate ``anchor`` inside ``page_text`` starting at ``after_char``.

    Tries exact → length-preserving punctuation-stripped → fuzzy match on
    the anchor's first 60 chars. Returns absolute (start, end) offsets or
    None.
    """
    clean = _sanitize_anchor(anchor)
    if not clean or not page_text:
        return None
    hay = page_text[after_char:]
    if not hay:
        return None

    m = re.search(re.escape(clean), hay, re.IGNORECASE)
    if m:
        return after_char + m.start(), after_char + m.end()

    norm_anchor = _normalize_for_match(clean)
    norm_hay = _normalize_for_match(hay)
    m = re.search(re.escape(norm_anchor), norm_hay)
    if m:
        return after_char + m.start(), after_char + m.end()

    probe = norm_anchor[:60]
    if len(probe) < 8:
        return None
    best_pos = -1
    best_ratio = 0.0
    step = max(1, len(probe) // 10)
    for i in range(0, max(1, len(norm_hay) - len(probe) // 2), step):
        window = norm_hay[i:i + len(probe)]
        ratio = SequenceMatcher(None, probe, window, autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_pos = i
    if best_pos >= 0 and best_ratio >= 0.7:
        return after_char + best_pos, after_char + best_pos + len(probe)
    return None


def _char_pos_to_word(
    char_pos: int,
    positions: list[tuple[int, int, int]],
    prefer_last: bool,
) -> int | None:
    """Map a char offset inside page_text to the originating word index."""
    if not positions:
        return None
    for cs, ce, wi in positions:
        if cs <= char_pos < ce:
            return wi
    if prefer_last:
        prev = None
        for cs, ce, wi in positions:
            if ce <= char_pos:
                prev = wi
            else:
                break
        return prev if prev is not None else positions[0][2]
    for cs, ce, wi in positions:
        if cs >= char_pos:
            return wi
    return positions[-1][2]


async def _find_section_by_text_anchors(
    node: dict,
    document_id: str,
    bbox_repo: OcrBboxRepository,
    img_w: float,
    img_h: float,
) -> tuple[list[dict], str]:
    """Tier 0: phrase-match start_text / end_text against the page OCR.

    Walks pages in page_range (±1 slack) to find the first start_text
    match, then continues forward looking for end_text — same page first,
    then subsequent pages. Emits one bbox per OCR block from start to end.

    Returns (bboxes, end_strategy). end_strategy is "end_text",
    "end_text:xpage", "marker" (fell back to next section number), or
    "range_end" (ran off the declared range without finding either).
    """
    start_text = node.get("start_text", "").strip()
    end_text = node.get("end_text", "").strip()
    page_range = node.get("page_range", [])
    if not start_text or not page_range:
        return [], ""

    declared_first = page_range[0]
    declared_last = page_range[-1] if len(page_range) >= 2 else page_range[0]
    slack_first = max(1, declared_first - 1)
    slack_last = declared_last + 1

    pages: list[tuple[int, list[dict], str, list[tuple[int, int, int]]]] = []
    for pg in range(slack_first, slack_last + 1):
        words = await bbox_repo.get_page_words(document_id, pg)
        if not words:
            continue
        page_text, positions = _build_page_text(words)
        pages.append((pg, words, page_text, positions))
    if not pages:
        return [], ""

    # Anchor on declared pages first; fall back to ±1 slack only if the
    # declared range has no match. Otherwise a fuzzy hit on neighbouring
    # boilerplate (e.g. दफा 161 near दफा 162) hijacks the citation.
    start_idx: int | None = None
    start_word_idx: int | None = None
    start_char_end = 0
    for in_declared in (True, False):
        for i, (pg, _words, page_text, positions) in enumerate(pages):
            is_declared = declared_first <= pg <= declared_last
            if is_declared != in_declared:
                continue
            span = _find_anchor_span(start_text, page_text)
            if span is not None:
                start_idx = i
                start_word_idx = _char_pos_to_word(span[0], positions, prefer_last=False)
                start_char_end = span[1]
                break
        if start_idx is not None:
            break
    if start_idx is None or start_word_idx is None:
        return [], ""

    end_idx: int | None = None
    end_word_idx: int | None = None
    end_strategy = ""

    if end_text:
        _pg, _words, page_text, positions = pages[start_idx]
        span = _find_anchor_span(end_text, page_text, after_char=start_char_end)
        if span is not None:
            end_idx = start_idx
            end_word_idx = _char_pos_to_word(span[1] - 1, positions, prefer_last=True)
            end_strategy = "end_text"
        else:
            for i in range(start_idx + 1, len(pages)):
                _pg, _words, page_text, positions = pages[i]
                span = _find_anchor_span(end_text, page_text)
                if span is not None:
                    end_idx = i
                    end_word_idx = _char_pos_to_word(span[1] - 1, positions, prefer_last=True)
                    end_strategy = "end_text:xpage"
                    break

    if end_idx is None:
        _pg, words, _page_text, positions = pages[start_idx]
        tail = [(cs, ce, wi) for cs, ce, wi in positions if cs >= start_char_end]
        marker_in_tail = None
        for idx_in_tail, (_cs, _ce, wi) in enumerate(tail):
            txt = str(words[wi].get("text", "") or "").strip()
            if re.match(r'^[०-९\d]+[\.:]', txt):
                marker_in_tail = idx_in_tail
                break
        if marker_in_tail is not None and marker_in_tail > 0:
            end_idx = start_idx
            end_word_idx = tail[marker_in_tail - 1][2]
            end_strategy = "marker"
        else:
            end_idx = len(pages) - 1
            _, _, _, end_positions = pages[end_idx]
            end_word_idx = end_positions[-1][2] if end_positions else None
            end_strategy = "range_end"

    if end_word_idx is None:
        return [], ""

    result: list[dict] = []
    for i in range(start_idx, end_idx + 1):
        pg, words, _page_text, positions = pages[i]
        ordered_idxs = [wi for _, _, wi in positions]
        if i == start_idx and i == end_idx:
            try:
                a = ordered_idxs.index(start_word_idx)
                b = ordered_idxs.index(end_word_idx)
            except ValueError:
                continue
            if b < a:
                return [], ""
            slice_ = ordered_idxs[a:b + 1]
        elif i == start_idx:
            try:
                a = ordered_idxs.index(start_word_idx)
            except ValueError:
                continue
            slice_ = ordered_idxs[a:]
        elif i == end_idx:
            try:
                b = ordered_idxs.index(end_word_idx)
            except ValueError:
                continue
            slice_ = ordered_idxs[:b + 1]
        else:
            slice_ = ordered_idxs

        for wi in slice_:
            w = words[wi]
            result.append({
                "page": pg,
                "bbox": {
                    "x0": w["x0"] * img_w,
                    "y0": w["y0"] * img_h,
                    "x2": w["x2"] * img_w,
                    "y2": w["y2"] * img_h,
                },
            })

    return result, end_strategy


async def _get_node_highlights(
    node: dict,
    document_id: str,
    bbox_repo: OcrBboxRepository,
    img_w: float,
    img_h: float,
    next_title: str | None = None,
) -> dict:
    """Build highlight entry for a single node.

    Strategy priority:
      0. Phrase match start_text / end_text directly on OCR words (primary —
         skips the markdown/char-offset indirection that fails when OCR text
         diverges from the parsed markdown). Cross-page aware.
      1. Char offset range (backup — relies on ingest-time offset alignment).
      2. Title y-search, bounded by next section-number marker.
      3. Raw section-level page_bboxes (last resort — one big rect per page).
    """
    page_bboxes: list[dict] = []
    node_page_bboxes = node.get("page_bboxes", [])
    start_char = node.get("start_char", -1)
    end_char = node.get("end_char", -1)

    strategy_used = "none"

    # Strategy 0 (primary): direct phrase match on OCR words.
    if node.get("start_text"):
        page_bboxes, end_strategy = await _find_section_by_text_anchors(
            node, document_id, bbox_repo, img_w, img_h,
        )
        if page_bboxes:
            strategy_used = f"text_anchors:{end_strategy}"

    # Strategy 1 (fallback): char offset word lookup.
    # When start_char/end_char are aligned (computed from start_text/end_text
    # at ingestion), this finds the exact OCR words at exact pixel positions.
    if not page_bboxes and start_char >= 0 and end_char > start_char:
        page_results = await bbox_repo.get_words_in_range(document_id, start_char, end_char)
        for page_data in page_results:
            for w in page_data["words"]:
                page_bboxes.append({
                    "page": page_data["page"],
                    "bbox": {
                        "x0": w["x0"] * img_w,
                        "y0": w["y0"] * img_h,
                        "x2": w["x2"] * img_w,
                        "y2": w["y2"] * img_h,
                    },
                })
        if page_bboxes:
            strategy_used = "char_range"

    # Strategy 2 (fallback): title search on target page
    if not page_bboxes:
        page_bboxes = await _find_section_by_title(node, document_id, bbox_repo, img_w, img_h, next_title)
        if page_bboxes:
            strategy_used = "title_search"

    # Strategy 3 (last resort): raw section-level page_bboxes
    if not page_bboxes:
        page_bboxes = node_page_bboxes
        if page_bboxes:
            strategy_used = "raw_page_bboxes"

    logger.info(
        f"Highlight node={node.get('nodeId','?')} | {strategy_used} | bboxes={len(page_bboxes)}"
    )

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
        next_title = _find_next_title(nodes, node_int_id)
        highlight = await _get_node_highlights(node, document_id, bbox_repo, img_w, img_h, next_title)
        all_highlights.append(highlight)

    return {
        "highlights": all_highlights,
        "image_dimensions": image_dimensions,
    }


@router.get("/{document_id}/page/{page_number}/words")
async def get_page_words(
    document_id: str,
    page_number: int,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return word-level OCR bboxes for a single page.

    Used by the frontend to render a transparent text-selection overlay on
    top of the page image (PDF.js-style drag-to-select + copy).

    Response shape:
        {
          "page": 1,
          "words": [
            {"text": "परिच्छेद-१", "x0": 0.12, "y0": 0.05, "x2": 0.38, "y2": 0.08,
             "line": 0, "block": 0, "char_offset": 0},
            ...
          ],
          "image_dimensions": {"width": 612, "height": 792}
        }

    Coordinates are normalized 0-1 relative to the page image. The frontend
    multiplies by its current rendered width/height to position each word.
    """
    tree_doc = await db.page_index_trees.find_one(
        {"document_id": document_id},
        {"image_dimensions": 1, "_id": 0},
    )
    if not tree_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    bbox_repo = OcrBboxRepository(db)
    words = await bbox_repo.get_page_words(document_id, page_number)

    return {
        "page": page_number,
        "words": words,
        "image_dimensions": tree_doc.get("image_dimensions"),
    }


@router.get("/{document_id}/words")
async def get_all_words(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_database),
):
    """Return word-level OCR bboxes for ALL pages in a single request.

    Preferred over ``/page/{page}/words`` for documents with many pages —
    saves N-1 round trips. Gzipped payload is typically 1-2 MB even for
    200+ page documents.

    Response shape:
        {
          "image_dimensions": {"width": 612, "height": 792},
          "words_by_page": {
            "1": [{"text": "...", "x0": ..., "y0": ..., "x2": ..., "y2": ..., ...}, ...],
            "2": [...],
            ...
          }
        }

    Coordinates are normalized 0-1 relative to the page image.
    """
    tree_doc = await db.page_index_trees.find_one(
        {"document_id": document_id},
        {"image_dimensions": 1, "_id": 0},
    )
    if not tree_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    bbox_repo = OcrBboxRepository(db)
    words_by_page = await bbox_repo.get_all_words(document_id)

    # JSON keys must be strings; MongoDB gave us int keys
    return {
        "image_dimensions": tree_doc.get("image_dimensions"),
        "words_by_page": {str(k): v for k, v in words_by_page.items()},
    }
