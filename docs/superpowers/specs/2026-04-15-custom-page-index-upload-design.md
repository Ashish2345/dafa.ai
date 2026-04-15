# Custom Page-Index Upload — Design

**Date:** 2026-04-15
**Status:** Draft for review

## Problem

Document ingestion currently generates the page-index tree (hierarchical sections with `nodeId`, `title`, `summary`, `page_range`) by running an LLM over the OCR'd markdown. This is expensive, slow, and not always best — users may already have a high-quality hand-curated or externally-generated index that they'd rather use directly.

The goal is to let a user supply their own tree JSON alongside the PDF, skipping the LLM tree-generation step while keeping everything else (OCR, markdown extraction, page bounding boxes, image storage) unchanged.

## Scope

**In scope:**
- Accept an optional JSON index file on the existing upload endpoint.
- Skip LLM tree generation when the file is supplied.
- Compute `start_char`/`end_char`, integer `id`, and `page_bboxes` on the supplied tree (same post-processing as the LLM path).
- Surface warnings when a supplied node's title can't be matched in the OCR'd markdown.

**Out of scope:**
- Replacing the tree on an already-ingested document (re-index flow).
- Supplying pre-computed `start_char`/`end_char` or `page_bboxes` — the system always recomputes these.
- Supporting custom trees with the `vector` retrieval strategy.
- Automated tests (per project preference).

## User-supplied JSON format

The supplied file matches the existing tree shape produced by the LLM:

```json
{
  "document_title": "string",
  "language": "en|ne",
  "nodes": [
    {
      "nodeId": "1.2.3",
      "title": "Section title",
      "summary": "Short summary",
      "page_range": [start_page, end_page],
      "children": [ /* recursive */ ]
    }
  ]
}
```

**Required per node:** `nodeId`, `title`, `summary`, `page_range`, `children`.

**System-computed (must not be supplied):** `id` (depth-first integer), `start_char`, `end_char`, `page_bboxes`. Any values the user supplies for these are ignored/overwritten.

## API

**Endpoint:** existing `POST /documents/upload` — no new route.

**New optional form field:** `page_index_file` (JSON file upload).

**Behavior matrix:**

| `page_index_file` | `strategy` | Result |
|---|---|---|
| absent | any | Current behavior (LLM generates tree for `page_index`; no tree for `vector`) |
| present | `page_index` | New: skip LLM, use provided tree |
| present | `vector` | Reject with 400 — `page_index_file` is only valid with `strategy=page_index` |

**Response:** unchanged — returns `{document_id, status: "processing"}` immediately. Schema validation of the JSON is synchronous; title-matching and warnings are async.

## Pipeline & strategy changes

### `app/services/ingestion/pipeline.py`
- `IngestionPipeline.run()` gains optional param `custom_tree: dict | None = None`.
- Passed through to `strategy.ingest(..., custom_tree=custom_tree)`.
- During the final document-save step, the pipeline persists `custom_tree_provided` and `ingest_warnings` on the document record (values returned by `strategy.ingest()`).

### `app/services/retrieval/page_index/strategy.py`
- `PageIndexStrategy.ingest()` gains optional param `custom_tree: dict | None = None`.
- The LLM-generation block (currently `split_chunks` → `build_chunk` loop → `_merge_trees`, lines ~140-154) is wrapped:
  ```
  if custom_tree is not None:
      tree = deepcopy(custom_tree)   # avoid mutating caller's dict
  else:
      # existing LLM path unchanged
  ```
- Post-processing runs identically in both paths: `_attach_char_offsets` → `_assign_node_ids` → `_attach_page_bboxes` → `save_tree`.
- Return value extended to include `ingest_warnings` (unmatched nodeIds). Pipeline consumes this for the document record.

### `app/services/retrieval/page_index/tree_builder.py`
- `_attach_char_offsets()` signature changes from `-> None` to `-> list[str]`, returning the list of nodeIds whose titles were not found in the markdown.
- Existing callers (LLM path) ignore the return value — backward compatible.
- Custom-tree path uses the list to populate `ingest_warnings`.

### `app/api/v1/endpoints/documents/routes.py`
- `page_index_file: Optional[UploadFile] = File(None)` added as a direct route parameter on the upload handler, alongside the existing `file: Optional[UploadFile]`. Note: it does NOT go on `ParseFormData` — that class uses `as_form()` to wrap fields in `Form()`, which doesn't work for file uploads. Files are passed directly as route parameters.
- Upload handler, synchronously:
  - If `page_index_file` is present + resolved strategy is `vector` → return 400.
  - If `page_index_file` is present, read bytes, parse as JSON, and validate via a Pydantic model (`PageIndexTreeUpload`). On any validation failure → return 400 with a clear message.
  - If parse + validation pass, pass the parsed dict to `_run_ingestion_background` via `custom_tree`.
- `_run_ingestion_background()` forwards `custom_tree` into `pipeline.run(..., custom_tree=custom_tree)`.

## Data model

**Document record (MongoDB `documents` collection) additions:**

```json
{
  "...existing fields...": "",
  "custom_tree_provided": true,
  "ingest_warnings": {
    "unmatched_nodes": [
      {"nodeId": "1.2", "title": "दफा २: परिभाषा (क-ज)"}
    ],
    "unmatched_count": 1,
    "total_nodes": 142
  }
}
```

- `custom_tree_provided`: `false` when the user didn't supply a tree (legacy LLM path).
- `ingest_warnings`: `null` when `custom_tree_provided` is `false` OR when all supplied titles matched. Otherwise populated as above.

**Status endpoint:** the existing endpoint that returns document state (to be identified during implementation — the document routes file already has a deletion and a listing; the status/detail endpoint for a single document is the target) includes `ingest_warnings` and `custom_tree_provided` in its response. This makes warnings durable (stored on the record) and visible on demand, not just during a polling window.

## Validation policy

**Synchronous (upload handler):**
- JSON is well-formed → else 400 `{"detail": "page_index_file is not valid JSON: <error>"}`.
- Pydantic schema: required fields present, types correct → else 400 with the offending field paths.
- Strategy compatibility: `strategy != vector` when `page_index_file` is supplied → else 400.

**Async (pipeline, non-fatal):**
- Node titles may not appear in the OCR'd markdown. Each such node is added to `ingest_warnings.unmatched_nodes`. Ingestion continues.
- Rationale: some titles will fail to match due to OCR variance (whitespace, punctuation glyphs like `ः` vs `:`). Rejecting the whole upload on one unmatched title would be too brittle. Silent-skip would hide degradation. Warnings let the user see exactly which nodes lost their precise char offsets and fix them.
- Nodes without char offsets fall back to regex-by-title, then summary-only text at query time. Retrieval still works; quality degrades on those nodes.

## Error handling

| Scenario | Where caught | Behavior |
|---|---|---|
| Malformed JSON | Upload handler (sync) | 400 with parse error |
| Missing required fields | Upload handler (sync) | 400 with offending paths |
| Invalid node shape | Upload handler (sync) | 400 with offending paths |
| `page_index_file` + `strategy=vector` | Upload handler (sync) | 400 with explanation |
| Node title not found in markdown | Background pipeline | Recorded in `ingest_warnings`; ingest continues |
| `page_range` out of PDF bounds | Background pipeline | Logged warning; `_attach_page_bboxes` already handles gracefully |
| Pipeline exception mid-ingest | Background task | Document status → `failed` (existing behavior) |

## Summary of files changed

- `app/api/v1/endpoints/documents/routes.py` — new form field, synchronous validation, strategy-compatibility check, thread `custom_tree` into pipeline.
- `app/services/ingestion/pipeline.py` — thread `custom_tree` parameter; persist `custom_tree_provided` + `ingest_warnings` on final save.
- `app/services/retrieval/page_index/strategy.py` — branch on `custom_tree` around the LLM loop; return warnings.
- `app/services/retrieval/page_index/tree_builder.py` — `_attach_char_offsets` returns list of unmatched nodeIds.
- `app/models/schemas.py` — new `PageIndexTreeUpload` + `PageIndexNodeUpload` Pydantic models for incoming JSON schema validation.
- `app/db/repositories/document_repository.py` — add method (or extend existing update method) to persist `custom_tree_provided` + `ingest_warnings` onto the document record.

No new endpoints, no new collections, no schema migration needed (MongoDB; new fields are additive and default to `null`/`false` on existing documents).
