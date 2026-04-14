# Citation Pills & Section Highlighting — Design Spec

**Date:** 2026-04-14
**Scope:** First phase of the reference/highlighting system for Mero Dafa
**Goal:** Let CAs verify AI-cited sections by clicking citation pills in the chat answer, which scroll the PDF viewer to the correct page and highlight the exact section on the page image.

---

## Context

Mero Dafa is a legal document search platform for Nepal. The backend ingests PDFs (acts, directives, gazette notices) via OCR, builds a PageIndex tree using an LLM, and answers legal queries with citation-backed responses. The frontend has a split-pane layout: chat on the left, page image viewer on the right.

Today, the AI answer includes section references as plain text. There is no way to click a citation and jump to the source on the page image. CAs must manually scroll and visually locate the referenced section — a trust gap for legal compliance work.

## What We're Building

An interactive citation system with three components:

1. **Citation pills** in the chat answer — clickable inline badges like `[Sec 2.1, Page 3]`
2. **Section highlight overlay** on page images — tight blue glow rectangles over the cited text
3. **Hover tooltip** — a compact preview card showing the Nepali section title and first few lines

## What We're NOT Building (Future Phases)

- Connector lines between chat and PDF (Leya-style "magic bridge")
- Source sidebar cards (NotebookLM-style)
- Bilingual sync highlighting (English chat ↔ Nepali PDF)
- "Verify with Gazette" trust stamp
- Word-level highlighting in the chat text

---

## Backend Changes

### 1. Persist Word-Level OCR Bounding Boxes

**Problem:** The OCR pipeline (Google Vision, digital OCR) returns word-level bounding boxes (`x0, y0, x2, y2` normalized 0-1), but these are discarded during markdown conversion. Only page-level aggregates are stored.

**Change:** After OCR processing, save the full word-level bbox data to a new MongoDB collection `page_ocr_bboxes`.

**Collection schema — one document per page** (avoids MongoDB's 16MB document size limit for large PDFs):

```json
{
  "document_id": "uuid-string",
  "page": 1,
  "words": [
    {
      "text": "दफा",
      "x0": 0.12,
      "y0": 0.05,
      "x2": 0.18,
      "y2": 0.08,
      "block": 0,
      "line": 0,
      "char_offset": 0
    }
  ]
}
```

**Index:** Compound index on `(document_id, page)` for fast lookups. Secondary index on `(document_id, words.char_offset)` for character-range queries across pages.

**Field definitions:**
- `x0, y0`: Top-left corner of word bounding box (normalized 0-1 relative to page dimensions)
- `x2, y2`: Bottom-right corner (normalized 0-1)
- `block`: OCR block number (groups of related paragraphs)
- `line`: Line number within the block
- `char_offset`: Character position of this word in the full markdown string — the bridge between `start_char`/`end_char` on tree nodes and pixel positions on the page image

**Where the change happens:**
- `app/services/parsers/ocr_input_parser/parser.py` — currently aggregates word bboxes to page-level at line ~421. Add a step to extract and return the word-level data before aggregation.
- `app/services/processing/markdown.py` — pass word-level bboxes through alongside the existing `page_bbox_map`
- `app/services/ingestion/pipeline.py` — persist via a new repository method
- New file: `app/db/repositories/ocr_bbox_repository.py` — handles CRUD for `page_ocr_bboxes` collection

**Storage estimate:** ~300 words/page × ~100 bytes ≈ ~30KB per page document. A 600-page PDF produces ~600 MongoDB documents totaling ~18MB — well within limits with per-page sharding.

**char_offset computation:** As the OCR DataFrame is converted to markdown (sequentially, page by page, block by block, line by line), track a running character counter. Each word's `char_offset` = the position in the full markdown where that word's text begins. This creates a direct mapping from any `start_char:end_char` range to the exact word bboxes within it.

### 2. New Endpoint: GET /api/v1/documents/{id}/highlights

**Purpose:** Given a document and a section identifier, return the word-level bounding boxes needed to render highlights on the page image.

**Request:**
```
GET /api/v1/documents/{doc_id}/highlights?node_id=2.1
```

Alternative parameter form (for flexibility):
```
GET /api/v1/documents/{doc_id}/highlights?start_char=1500&end_char=3200
```

**Response:**
```json
{
  "document_id": "uuid",
  "node_id": "2.1",
  "highlights": [
    {
      "page": 3,
      "lines": [
        {
          "line": 0,
          "block": 2,
          "bbox": {"x0": 0.10, "y0": 0.18, "x2": 0.85, "y2": 0.22},
          "words": [
            {"text": "दफा", "x0": 0.10, "y0": 0.18, "x2": 0.16, "y2": 0.22},
            {"text": "२.१", "x0": 0.17, "y0": 0.18, "x2": 0.22, "y2": 0.22}
          ]
        },
        {
          "line": 1,
          "block": 2,
          "bbox": {"x0": 0.10, "y0": 0.23, "x2": 0.90, "y2": 0.27},
          "words": [ ... ]
        }
      ]
    }
  ]
}
```

**Authentication:** Same auth middleware as other endpoints — requires valid JWT token.

**How it works:**
1. Look up the tree node by `node_id` to get `start_char` and `end_char`
2. Query `page_ocr_bboxes` for all page documents matching `document_id`, then filter words where `char_offset >= start_char AND char_offset < end_char`
3. Group words by page, then by line
4. Compute a per-line bounding box (min x0, min y0, max x2, max y2 of words in that line)
5. Return grouped result

**Performance:** Compound index on `(document_id, page)` plus filtering words by `char_offset` range. Since each page document is small (~30KB), even scanning all pages for a document and filtering in application code is fast.

### 3. Include Section-Level page_bboxes in Query Response (Fallback)

**Purpose:** For documents ingested before word-level bbox storage is deployed, provide approximate highlights using the existing section-level `page_bboxes` already stored in `page_index_trees`.

**Change:** In `app/services/retrieval/page_index/strategy.py`, when building the `RetrievedChunk`, include the node's `page_bboxes` in the `source` dict:

```python
"source": {
    "document_name": ...,
    "document_id": ...,
    "page_range": ...,
    "page": ...,
    "section": ...,
    "node_id": ...,
    "node_int_id": ...,
    "page_bboxes": node.get("page_bboxes", [])  # NEW
}
```

**Frontend behavior:** If the highlights endpoint returns data, use word-level bboxes. If it returns empty (old document), fall back to `page_bboxes` from the query response for approximate section-level highlighting.

### 4. LLM Prompt Update

**Change:** Update the answer synthesis prompt to instruct the LLM to wrap citations in structured HTML `<cite>` tags.

**Current prompt pattern** (from flow.md):
```
System: "Use ONLY the provided sections. Cite (Section nodeId: title)..."
```

**New instruction added to system prompt:**
```
When citing a source, wrap it in a <cite> tag with data attributes:
<cite data-node="{node_id}" data-doc="{document_id}" data-page="{first_page}" data-section="{section_title}">{display text}</cite>

Example: <cite data-node="2.1" data-doc="4a3efdb0-281f-4d14-96cc-0ca844a687f0" data-page="3" data-section="Remuneration Payments">Section 2.1, Page 3</cite>
```

**Why structured HTML:** The LLM already returns HTML. `data-*` attributes carry machine-readable IDs without polluting display text. The frontend doesn't need regex parsing — just query `cite[data-node]` elements.

**Data availability:** The chunks passed to the LLM for synthesis already contain `document_id`, `node_id`, `page_range`, and `section` title. The prompt just needs to tell the LLM to use them.

---

## Frontend Changes

### 1. Citation Pill Component

**What:** Parse `<cite>` tags from the LLM HTML response and render them as interactive inline badges.

**Visual style:**
- Rounded chip: `border-radius: 12px`
- Background: `rgba(37, 99, 235, 0.2)`
- Text color: `#60a5fa` (light blue)
- Border: `1px solid rgba(37, 99, 235, 0.3)`
- Small page icon (SVG) before the text
- Cursor: pointer

**Active state (after click):**
- Background: `rgba(37, 99, 235, 0.35)`
- Border: `1px solid rgba(37, 99, 235, 0.6)`

**Implementation:** Since the LLM returns HTML rendered directly, the frontend needs an event delegation handler on the chat container that listens for clicks/hovers on `cite[data-node]` elements. No special React component parsing required — just CSS styling for `cite` tags and JS event handlers.

### 2. Highlight Overlay System

**What:** Render transparent highlight rectangles on top of page images to show the cited section.

**Structure:**
```
<div style="position: relative">         <!-- wrapper -->
  <img src="/api/v1/documents/{id}/image/{page}" />  <!-- page image -->
  <svg style="position: absolute; inset: 0">         <!-- overlay layer -->
    <rect x="10%" y="18%" width="75%" height="4%" ... />  <!-- per-line highlight -->
    <rect x="10%" y="23%" width="80%" height="4%" ... />
  </svg>
</div>
```

**Why SVG over divs:** SVG handles percentage-based positioning natively and scales cleanly with the image. Multiple rectangles in one SVG layer is cleaner than multiple absolutely positioned divs.

**Visual style per rectangle:**
- Fill: `rgba(37, 99, 235, 0.15)` — royal blue at 15% opacity
- Stroke: `rgba(37, 99, 235, 0.4)`, 2px
- Filter: `drop-shadow(0 0 8px rgba(37, 99, 235, 0.25))` — the "searchlight" glow

**Coordinates:** All bbox values are normalized 0-1. Convert to SVG percentages: `x="x0 * 100%"`, `y="y0 * 100%"`, `width="(x2 - x0) * 100%"`, `height="(y2 - y0) * 100%"`. The SVG viewBox matches the image aspect ratio, so highlights scale with zoom.

### 3. Scroll + Pulse Interaction

**On citation pill click:**

1. Extract `data-doc`, `data-node`, `data-page` from the `<cite>` element
2. Call `GET /api/v1/documents/{doc_id}/highlights?node_id={node_id}`
3. Clear any existing highlights from previous citation
4. Smooth-scroll the right pane to the target page image
5. Render highlight rectangles on the page image(s)
6. Play pulse animation: 3 flashes over ~1.2 seconds (opacity oscillates 1.0 → 0.3 → 1.0)
7. After pulse, keep highlights visible with persistent active border
8. Set the clicked citation pill to active state

**On clicking another citation:** Clear previous highlights and active pill state, activate new one.

**On clicking the same citation again:** Toggle off — remove highlights and deactivate pill.

### 4. Hover Tooltip

**On hovering a citation pill (200ms delay):**

A compact card appears anchored near the pill:
- **Line 1:** Section title in Nepali (from `data-section` attribute or the chunk's `text` field)
- **Line 2-3:** First ~150 characters of the section's text (already available in the query response chunk `text` field)
- **Line 4:** Small muted text: page number and document name

**Style:**
- Dark background: `#1e293b`
- Border: `1px solid rgba(37, 99, 235, 0.3)`
- Border radius: 8px
- Shadow: `0 8px 24px rgba(0, 0, 0, 0.3)`
- Max width: 280px

**Behavior:**
- Appears on hover after 200ms delay
- Disappears on mouse-out
- If the user clicks instead of hovering, skip tooltip and go straight to scroll + highlight

**Data source:** The tooltip content comes from the query response chunks already in frontend memory. No additional API call needed.

---

## Data Flow Summary

```
User asks question
  → POST /query
  → Backend retrieves chunks (with node_id, page_range, text, page_bboxes)
  → LLM synthesizes answer with <cite> tags
  → Frontend renders answer HTML with styled citation pills

User clicks citation pill [Sec 2.1, Page 3]
  → Frontend calls GET /documents/{doc_id}/highlights?node_id=2.1
  → Backend looks up node's start_char/end_char
  → Backend queries page_ocr_bboxes for words in that char range
  → Backend returns word bboxes grouped by page and line
  → Frontend smooth-scrolls right pane to page 3
  → Frontend renders SVG highlight rectangles over the section
  → Pulse animation plays, then persistent active state

User hovers citation pill
  → Frontend shows tooltip with chunk text (already in memory)
  → No API call
```

---

## Fallback Behavior

| Scenario | Highlight source |
|----------|-----------------|
| Document ingested after this feature ships | Word-level bboxes from `page_ocr_bboxes` → tight per-line highlights |
| Document ingested before this feature | Section-level `page_bboxes` from query response → approximate rectangular overlay |
| Highlights endpoint returns empty | Page-level scroll only (no overlay), citation pill still works for navigation |

---

## Migration

Already-ingested documents will not have word-level bboxes in `page_ocr_bboxes`. Two options:

1. **Lazy re-ingestion:** Add a management script or background task that re-runs OCR bbox extraction for existing documents without re-running the full ingestion pipeline. Priority documents (frequently queried acts) first.
2. **Fallback gracefully:** The section-level `page_bboxes` already stored in `page_index_trees` provide approximate highlights. Not as precise, but functional.

Both approaches should be implemented: fallback for immediate coverage, lazy re-ingestion for precision over time.

---

## Files Affected

**Backend (new files):**
- `app/db/repositories/ocr_bbox_repository.py` — CRUD for `page_ocr_bboxes` collection
- `app/api/v1/endpoints/highlights.py` — GET highlights endpoint

**Backend (modified files):**
- `app/services/parsers/ocr_input_parser/parser.py` — extract word-level bboxes with char_offset before aggregation
- `app/services/processing/markdown.py` — pass word-level bboxes through the pipeline
- `app/services/ingestion/pipeline.py` — persist word-level bboxes via new repository
- `app/services/retrieval/page_index/strategy.py` — include `page_bboxes` in query response chunks
- `app/prompts/` — update synthesis prompt to output `<cite>` tags
- `app/api/v1/router.py` — register highlights endpoint

**Frontend (changes):**
- Chat answer renderer — CSS for `cite` tag styling + event handlers for click/hover
- Page image viewer — SVG overlay layer for highlights, scroll-to-page logic
- Tooltip component — hover preview card
- API client — new `getHighlights()` function
