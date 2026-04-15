# Custom Page-Index Upload — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user POST an optional `page_index_file` (hierarchical tree JSON) to `/documents/upload` alongside the PDF. When supplied, the pipeline skips LLM tree generation and uses the provided tree, while still computing `start_char`/`end_char`, integer `id`, and `page_bboxes` against the OCR'd markdown.

**Architecture:** Single thread-through of `custom_tree: dict | None` from the upload route → `IngestionPipeline.run()` → `PageIndexStrategy.ingest()`. Inside the strategy, branch around the LLM loop; post-processing runs unchanged. Unmatched node titles surface as non-fatal warnings persisted on the document record.

**Tech Stack:** FastAPI, Pydantic v2, MongoDB (motor async), Python 3.11+

**Spec:** [docs/superpowers/specs/2026-04-15-custom-page-index-upload-design.md](../specs/2026-04-15-custom-page-index-upload-design.md)

**Note:** Per project convention, this plan has no automated test tasks. Each task ends with a lightweight verification step (import check or syntax check) plus a commit. A final manual end-to-end curl smoke test lives in Task 8.

---

## File Structure

**Modified files (in order of dependency):**
- `app/models/schemas.py` — add `PageIndexNodeUpload` + `PageIndexTreeUpload` Pydantic models
- `app/services/retrieval/page_index/tree_builder.py` — `_attach_char_offsets` returns unmatched nodeIds
- `app/services/retrieval/page_index/strategy.py` — `ingest()` accepts `custom_tree`, returns warnings dict
- `app/db/repositories/document_repository.py` — extend `save()` to persist `custom_tree_provided` + `ingest_warnings`
- `app/services/ingestion/pipeline.py` — thread `custom_tree` through; capture warnings and pass to repo
- `app/api/v1/endpoints/documents/routes.py` — new `page_index_file` route parameter, synchronous JSON validation, strategy-compat check, thread `custom_tree` into background task

**No new endpoints, no new collections, no schema migrations** (MongoDB; new fields are additive).

---

### Task 1: Pydantic models for upload validation

**Files:**
- Modify: `app/models/schemas.py` (append new classes at end of file)

- [ ] **Step 1: Read the existing file to find a good insertion point**

Run: Read `app/models/schemas.py` to see existing models and the import section.

- [ ] **Step 2: Add the models at the bottom of the file**

Append to `app/models/schemas.py`:

```python
# ---------------------------------------------------------------------------
# Custom page-index tree upload (used by POST /documents/upload when the user
# supplies their own tree instead of letting the LLM generate one).
# ---------------------------------------------------------------------------

from __future__ import annotations  # only if not already imported at top

from pydantic import BaseModel, Field, field_validator


class PageIndexNodeUpload(BaseModel):
    """One node in a user-supplied page-index tree.

    System-computed fields (``id``, ``start_char``, ``end_char``, ``page_bboxes``)
    are intentionally *not* present on this model — they are ignored/overwritten
    by the ingestion pipeline even if supplied.
    """

    nodeId: str = Field(..., min_length=1, description="Dotted id like '1.2.3'")
    title: str = Field(..., min_length=1)
    summary: str = Field(default="", description="Short summary; may be empty")
    page_range: list[int] = Field(..., description="[start_page, end_page], both 1-indexed")
    children: list["PageIndexNodeUpload"] = Field(default_factory=list)

    @field_validator("page_range")
    @classmethod
    def _check_page_range(cls, v: list[int]) -> list[int]:
        if len(v) != 2:
            raise ValueError("page_range must be a 2-element list [start, end]")
        start, end = v
        if start < 1 or end < 1:
            raise ValueError("page_range values must be >= 1 (pages are 1-indexed)")
        if start > end:
            raise ValueError(f"page_range start ({start}) cannot exceed end ({end})")
        return v


class PageIndexTreeUpload(BaseModel):
    """Top-level shape of a user-supplied page-index tree JSON file."""

    document_title: str = Field(..., min_length=1)
    language: str = Field(default="en", description="e.g. 'en' or 'ne'")
    nodes: list[PageIndexNodeUpload] = Field(..., min_length=1)


PageIndexNodeUpload.model_rebuild()
```

Notes:
- If the file already imports `from __future__ import annotations`, skip that line; only add what's missing.
- If `BaseModel`/`Field` are already imported at the top of the file, do not duplicate imports — only add `field_validator` if not present.
- `model_rebuild()` is needed because the model is self-referential via the string `"PageIndexNodeUpload"`.

- [ ] **Step 3: Verify the module imports cleanly**

Run: `cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && python -c "from app.models.schemas import PageIndexTreeUpload, PageIndexNodeUpload; print('ok')"`

Expected output: `ok`

- [ ] **Step 4: Spot-check validation with a short REPL snippet**

Run:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && python -c "
from app.models.schemas import PageIndexTreeUpload
tree = PageIndexTreeUpload.model_validate({
    'document_title': 'Test',
    'language': 'ne',
    'nodes': [{'nodeId': '1', 'title': 'A', 'summary': 's', 'page_range': [1, 2], 'children': []}],
})
print('ok:', tree.document_title, len(tree.nodes))
"
```

Expected output: `ok: Test 1`

Also verify a bad page_range is rejected:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && python -c "
from app.models.schemas import PageIndexTreeUpload
from pydantic import ValidationError
try:
    PageIndexTreeUpload.model_validate({
        'document_title': 'Test',
        'nodes': [{'nodeId': '1', 'title': 'A', 'summary': '', 'page_range': [5, 2], 'children': []}],
    })
    print('FAIL: should have raised')
except ValidationError as e:
    print('ok: rejected')
"
```

Expected output: `ok: rejected`

- [ ] **Step 5: Commit**

```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  git add app/models/schemas.py && \
  git commit -m "feat(schemas): add PageIndexTreeUpload model for custom tree upload"
```

---

### Task 2: `_attach_char_offsets` returns unmatched nodeIds

**Files:**
- Modify: `app/services/retrieval/page_index/tree_builder.py` (lines 199-223 and callers)

The current `_attach_char_offsets` returns `None` and silently sets `start_char = -1` for nodes whose titles aren't found in the markdown. We need to collect those nodeIds and return them so the custom-tree path can report warnings.

- [ ] **Step 1: Change `_attach_char_offsets` signature and body**

Find this block in `app/services/retrieval/page_index/tree_builder.py` (around line 199):

```python
    def _attach_char_offsets(
        self, nodes: list, markdown: str, page_bbox_map: list[dict] | None = None,
    ) -> None:
        """Record start_char/end_char for each node so text extraction is a simple slice.
        ...
        """
        # Build page → char range lookup for constraining searches
        page_char_ranges: dict[int, tuple[int, int]] = {}
        if page_bbox_map:
            for entry in page_bbox_map:
                page_char_ranges[entry["page"]] = (entry["start_char"], entry["end_char"])

        # Pass 1: assign start_char to each node
        self._assign_start_chars(nodes, markdown, page_char_ranges)

        # Pass 2: collect all start positions, then assign end_char
        all_starts = sorted(set(self._collect_start_chars(nodes)))
        self._assign_end_chars(nodes, markdown, all_starts)
```

Replace with:

```python
    def _attach_char_offsets(
        self, nodes: list, markdown: str, page_bbox_map: list[dict] | None = None,
    ) -> list[dict]:
        """Record start_char/end_char for each node so text extraction is a simple slice.

        Two-pass approach:
          Pass 1: Assign start_char to every node by searching for its title in
                  the markdown, constrained to the node's page_range region.
          Pass 2: Collect all assigned start_chars as boundary positions, then
                  set each node's end_char to the next boundary (or EOF).

        Returns:
            A list of ``{"nodeId": ..., "title": ...}`` dicts for every node
            whose title was NOT found in the markdown (``start_char == -1``).
            The LLM-generated path ignores the return value; the custom-tree
            path uses it to populate ``ingest_warnings``.
        """
        # Build page → char range lookup for constraining searches
        page_char_ranges: dict[int, tuple[int, int]] = {}
        if page_bbox_map:
            for entry in page_bbox_map:
                page_char_ranges[entry["page"]] = (entry["start_char"], entry["end_char"])

        # Pass 1: assign start_char to each node
        self._assign_start_chars(nodes, markdown, page_char_ranges)

        # Pass 2: collect all start positions, then assign end_char
        all_starts = sorted(set(self._collect_start_chars(nodes)))
        self._assign_end_chars(nodes, markdown, all_starts)

        # Pass 3: collect nodeIds whose titles were not found
        return self._collect_unmatched(nodes)
```

- [ ] **Step 2: Add the `_collect_unmatched` helper**

Append this method to the `TreeBuilder` class (e.g., right after `_collect_start_chars` around line 297):

```python
    def _collect_unmatched(self, nodes: list) -> list[dict]:
        """Recursively collect ``{nodeId, title}`` for nodes with start_char == -1."""
        result: list[dict] = []
        for node in nodes:
            if node.get("start_char", -1) < 0:
                result.append({
                    "nodeId": node.get("nodeId", ""),
                    "title": node.get("title", ""),
                })
            result.extend(self._collect_unmatched(node.get("children", [])))
        return result
```

- [ ] **Step 3: Verify the module still imports cleanly**

Run:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  python -c "from app.services.retrieval.page_index.tree_builder import TreeBuilder; tb = TreeBuilder.__new__(TreeBuilder); print('ok')"
```

Expected output: `ok`

- [ ] **Step 4: Verify existing LLM callers still work**

The only caller that passes through the LLM path is `PageIndexStrategy.ingest()` at line 157:
```python
self.tree_builder._attach_char_offsets(tree.get("nodes", []), markdown, page_bbox_map)
```
It already ignores the return value (calls the method as a statement), so the `None` → `list[dict]` change is backward compatible. Confirm by grepping for other callers:

Run:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  grep -rn "_attach_char_offsets" --include="*.py" app/ tests/ 2>/dev/null
```

Expected: only `strategy.py` and the definition in `tree_builder.py` should appear. If any other caller exists, confirm it's still compatible with the new return type (discarding a list instead of None is safe).

- [ ] **Step 5: Commit**

```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  git add app/services/retrieval/page_index/tree_builder.py && \
  git commit -m "refactor(tree_builder): return unmatched nodeIds from _attach_char_offsets"
```

---

### Task 3: `PageIndexStrategy.ingest()` accepts `custom_tree`

**Files:**
- Modify: `app/services/retrieval/page_index/strategy.py` (lines 117-191)

- [ ] **Step 1: Add `copy.deepcopy` import if not already present**

Check the imports at the top of `app/services/retrieval/page_index/strategy.py`. If `deepcopy` isn't imported, add at the top-level imports:

```python
from copy import deepcopy
```

- [ ] **Step 2: Extend the `ingest()` signature and branch around the LLM loop**

Find the existing `ingest()` method starting around line 117:

```python
    async def ingest(
        self,
        document_id: str,
        markdown: str,
        metadata: dict[str, Any],
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
        page_bbox_map: list[dict] | None = None,
        image_dimensions: dict | None = None,
    ) -> None:
```

Change the signature and return type to:

```python
    async def ingest(
        self,
        document_id: str,
        markdown: str,
        metadata: dict[str, Any],
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
        page_bbox_map: list[dict] | None = None,
        image_dimensions: dict | None = None,
        custom_tree: dict | None = None,
    ) -> dict:
```

Then find this block (currently lines ~139-154):

```python
        # Split here so we can report async progress between chunks
        chunks = self.tree_builder.split_chunks(markdown)
        total = len(chunks)

        subtrees = []
        for i, chunk in enumerate(chunks, 1):
            await _notify(f"Building index tree: chunk {i}/{total} ({len(chunk):,} chars)...")
            # Run the blocking LLM call in a thread pool so the event loop
            # stays alive for MongoDB keepalives during the 1-2 min API call.
            subtree = await asyncio.to_thread(self.tree_builder.build_chunk, chunk, language)
            subtrees.append(subtree)

        if total == 1:
            tree = subtrees[0]
        else:
            tree = self.tree_builder._merge_trees(subtrees, language)
```

Replace with:

```python
        if custom_tree is not None:
            await _notify("Using user-supplied page-index tree (skipping LLM generation)...")
            # deepcopy so we don't mutate the caller's dict (it gets written to
            # with start_char/end_char/id/page_bboxes by the post-processing steps).
            tree = deepcopy(custom_tree)
        else:
            # Split here so we can report async progress between chunks
            chunks = self.tree_builder.split_chunks(markdown)
            total = len(chunks)

            subtrees = []
            for i, chunk in enumerate(chunks, 1):
                await _notify(f"Building index tree: chunk {i}/{total} ({len(chunk):,} chars)...")
                # Run the blocking LLM call in a thread pool so the event loop
                # stays alive for MongoDB keepalives during the 1-2 min API call.
                subtree = await asyncio.to_thread(self.tree_builder.build_chunk, chunk, language)
                subtrees.append(subtree)

            if total == 1:
                tree = subtrees[0]
            else:
                tree = self.tree_builder._merge_trees(subtrees, language)
```

- [ ] **Step 3: Capture unmatched nodes and build warnings dict**

Find this line (currently line 157):

```python
        self.tree_builder._attach_char_offsets(tree.get("nodes", []), markdown, page_bbox_map)
```

Replace with:

```python
        unmatched_nodes = self.tree_builder._attach_char_offsets(
            tree.get("nodes", []), markdown, page_bbox_map
        )
```

- [ ] **Step 4: Build the return value**

Find the end of the method, which currently ends like:

```python
        await _notify(f"PageIndex tree saved ({node_count} nodes)")
```

Add this helper and new return just before `async def _filter_by_collection`:

Immediately after `await _notify(f"PageIndex tree saved ({node_count} nodes)")` (the last line of the ingest method body), insert:

```python

        # Build warnings payload (only meaningful for the custom-tree path —
        # LLM-generated trees have start_char assigned for every match the LLM
        # produced; unmatched_nodes there just reflects titles that drifted
        # during OCR and is not actionable. For the LLM path, we still compute
        # it so the pipeline can log it, but we do not surface it as a warning.)
        if custom_tree is not None:
            ingest_warnings = {
                "unmatched_nodes": unmatched_nodes,
                "unmatched_count": len(unmatched_nodes),
                "total_nodes": node_count,
            } if unmatched_nodes else None
        else:
            ingest_warnings = None

        return {
            "custom_tree_provided": custom_tree is not None,
            "ingest_warnings": ingest_warnings,
            "node_count": node_count,
        }
```

- [ ] **Step 5: Verify the module imports cleanly**

Run:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  python -c "from app.services.retrieval.page_index.strategy import PageIndexStrategy; print('ok')"
```

Expected output: `ok`

- [ ] **Step 6: Verify signature by inspection**

Run:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  python -c "
import inspect
from app.services.retrieval.page_index.strategy import PageIndexStrategy
sig = inspect.signature(PageIndexStrategy.ingest)
assert 'custom_tree' in sig.parameters, f'custom_tree missing from {sig}'
assert sig.parameters['custom_tree'].default is None
print('ok:', sig)
"
```

Expected output: `ok: (self, document_id: str, markdown: str, metadata: dict[str, typing.Any], on_progress: ..., page_bbox_map: ... = None, image_dimensions: ... = None, custom_tree: ... = None) -> dict`

- [ ] **Step 7: Commit**

```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  git add app/services/retrieval/page_index/strategy.py && \
  git commit -m "feat(page_index): accept custom_tree in ingest() and return warnings"
```

---

### Task 4: Extend `DocumentRepository.save` to persist warnings

**Files:**
- Modify: `app/db/repositories/document_repository.py` (the `save` method)

- [ ] **Step 1: Add new optional parameters to `save`**

Find the `save` method around line 53:

```python
    async def save(
        self, document_id: str, filename: str, metadata: dict[str, Any], strategy: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "document_id": document_id,
                    "filename": filename,
                    "metadata": metadata,
                    "strategy": strategy,
                    "status": "completed",
                    "progress_step": "Done",
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
```

Replace with:

```python
    async def save(
        self,
        document_id: str,
        filename: str,
        metadata: dict[str, Any],
        strategy: str,
        custom_tree_provided: bool = False,
        ingest_warnings: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "document_id": document_id,
                    "filename": filename,
                    "metadata": metadata,
                    "strategy": strategy,
                    "status": "completed",
                    "progress_step": "Done",
                    "custom_tree_provided": custom_tree_provided,
                    "ingest_warnings": ingest_warnings,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
```

Existing callers that don't pass the new args retain today's behavior (`custom_tree_provided: False`, `ingest_warnings: None`).

- [ ] **Step 2: Verify module imports cleanly**

Run:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  python -c "
import inspect
from app.db.repositories.document_repository import DocumentRepository
sig = inspect.signature(DocumentRepository.save)
assert 'custom_tree_provided' in sig.parameters
assert 'ingest_warnings' in sig.parameters
print('ok:', list(sig.parameters))
"
```

Expected output includes `'custom_tree_provided'` and `'ingest_warnings'`.

- [ ] **Step 3: Commit**

```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  git add app/db/repositories/document_repository.py && \
  git commit -m "feat(document_repo): persist custom_tree_provided and ingest_warnings"
```

---

### Task 5: Thread `custom_tree` through `IngestionPipeline.run()`

**Files:**
- Modify: `app/services/ingestion/pipeline.py` (lines 35-182)

- [ ] **Step 1: Extend the `run` signature**

Find the `run` method around line 35:

```python
    async def run(
        self,
        file_path: Path,
        filename: str,
        document_id: str,
        strategy: RetrievalStrategy,
        language: str = "en",
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> dict[str, Any]:
```

Add the new parameter:

```python
    async def run(
        self,
        file_path: Path,
        filename: str,
        document_id: str,
        strategy: RetrievalStrategy,
        language: str = "en",
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
        custom_tree: dict | None = None,
    ) -> dict[str, Any]:
```

- [ ] **Step 2: Forward `custom_tree` to `strategy.ingest()` and capture its return**

Find the block around line 155-164:

```python
            # Step 5: Strategy-specific storage (passes on_progress so chunked
            # strategies can report per-chunk status)
            await strategy.ingest(
                document_id,
                markdown,
                metadata,
                on_progress=on_progress,
                page_bbox_map=page_bbox_map,
                image_dimensions=image_dimensions,
            )
```

Replace with:

```python
            # Step 5: Strategy-specific storage (passes on_progress so chunked
            # strategies can report per-chunk status)
            ingest_kwargs = {
                "on_progress": on_progress,
                "page_bbox_map": page_bbox_map,
                "image_dimensions": image_dimensions,
            }
            # Only PageIndexStrategy accepts custom_tree; VectorStrategy does not.
            # The route handler guarantees custom_tree is only set when strategy
            # is PageIndex, but we still avoid passing the kwarg when it's None
            # so we stay compatible with strategies that don't expose it.
            if custom_tree is not None:
                ingest_kwargs["custom_tree"] = custom_tree

            strategy_result = await strategy.ingest(
                document_id,
                markdown,
                metadata,
                **ingest_kwargs,
            )
            # Back-compat: VectorStrategy.ingest() currently returns None.
            if not isinstance(strategy_result, dict):
                strategy_result = {}
```

- [ ] **Step 3: Persist warnings via the document repo**

Find the block around line 166-174:

```python
            # Step 6: Record in document repository
            await _step("Saving document record...")
            if self.document_repo:
                await self.document_repo.save(
                    document_id=document_id,
                    filename=filename,
                    metadata=metadata,
                    strategy=strategy.__class__.__name__,
                )
```

Replace with:

```python
            # Step 6: Record in document repository
            await _step("Saving document record...")
            if self.document_repo:
                await self.document_repo.save(
                    document_id=document_id,
                    filename=filename,
                    metadata=metadata,
                    strategy=strategy.__class__.__name__,
                    custom_tree_provided=bool(strategy_result.get("custom_tree_provided", False)),
                    ingest_warnings=strategy_result.get("ingest_warnings"),
                )
```

- [ ] **Step 4: Verify the module imports cleanly**

Run:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  python -c "
import inspect
from app.services.ingestion.pipeline import IngestionPipeline
sig = inspect.signature(IngestionPipeline.run)
assert 'custom_tree' in sig.parameters
print('ok')
"
```

Expected output: `ok`

- [ ] **Step 5: Commit**

```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  git add app/services/ingestion/pipeline.py && \
  git commit -m "feat(pipeline): thread custom_tree through run() and persist warnings"
```

---

### Task 6: Accept `page_index_file` on the upload route

**Files:**
- Modify: `app/api/v1/endpoints/documents/routes.py` (the `upload_document` handler and `_run_ingestion_background`)

- [ ] **Step 1: Add imports**

Near the top of `app/api/v1/endpoints/documents/routes.py`, ensure these are present (add what's missing):

```python
import json
from pydantic import ValidationError

from app.models.schemas import PageIndexTreeUpload
```

Leave existing imports untouched.

- [ ] **Step 2: Add `page_index_file` parameter to the upload route**

Find the `upload_document` handler signature (around line 102):

```python
@router.post("/upload", summary="Upload and ingest a document", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    form_data: ParseFormData = Depends(ParseFormData.as_form()),
    current_user: dict = Depends(get_current_user),
):
```

Replace with:

```python
@router.post("/upload", summary="Upload and ingest a document", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    page_index_file: Optional[UploadFile] = File(
        None,
        description=(
            "Optional user-supplied page-index tree JSON. When provided, the "
            "pipeline skips LLM tree generation and uses this tree instead. "
            "Only valid with strategy='page_index'."
        ),
    ),
    form_data: ParseFormData = Depends(ParseFormData.as_form()),
    current_user: dict = Depends(get_current_user),
):
```

- [ ] **Step 3: Add synchronous validation right after filename extraction**

Find the existing block in `upload_document` (around lines 114-122):

```python
    original_filename = None
    if file and file.filename:
        original_filename = file.filename
    elif form_data.file_url:
        from urllib.parse import urlparse
        parsed_url = urlparse(form_data.file_url)
        if parsed_url.path:
            original_filename = Path(parsed_url.path).name
```

**Immediately after that block**, insert:

```python
    # ------------------------------------------------------------------
    # Custom page-index tree: parse and validate synchronously.
    # If invalid, fail with 400 before starting the slow OCR background task.
    # ------------------------------------------------------------------
    custom_tree: Optional[dict] = None
    if page_index_file is not None:
        # Strategy compatibility check: only page_index supports custom trees.
        strategy_name = (getattr(form_data, "strategy", None)
                         or "page_index").lower()
        if strategy_name == "vector":
            raise HTTPException(
                status_code=400,
                detail="page_index_file is only valid with strategy='page_index'.",
            )

        raw = await page_index_file.read()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"page_index_file is not valid JSON: {exc.msg} at line {exc.lineno} col {exc.colno}",
            ) from exc

        try:
            tree_model = PageIndexTreeUpload.model_validate(parsed)
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "page_index_file failed schema validation",
                    "errors": exc.errors(),
                },
            ) from exc

        # Use the parsed dict (not the model) for downstream flexibility — the
        # strategy expects a plain nested dict shape matching the LLM output.
        custom_tree = tree_model.model_dump()
```

- [ ] **Step 4: Pass `custom_tree` to the background task**

Find the `background_tasks.add_task(...)` call (around line 143):

```python
        # Hand off to background — temp file is deleted by the task when done
        background_tasks.add_task(
            _run_ingestion_background,
            file_path,
            document_id,
            filename,
            form_data,
        )
```

Replace with:

```python
        # Hand off to background — temp file is deleted by the task when done
        background_tasks.add_task(
            _run_ingestion_background,
            file_path,
            document_id,
            filename,
            form_data,
            custom_tree,
        )
```

- [ ] **Step 5: Update `_run_ingestion_background` to accept and forward `custom_tree`**

Find the function signature (around line 38):

```python
async def _run_ingestion_background(
    file_path: str,
    document_id: str,
    filename: str,
    form_data: ParseFormData,
) -> None:
```

Replace with:

```python
async def _run_ingestion_background(
    file_path: str,
    document_id: str,
    filename: str,
    form_data: ParseFormData,
    custom_tree: Optional[dict] = None,
) -> None:
```

Then find the `pipeline.run(...)` call (around line 70):

```python
        result = await pipeline.run(
            file_path=Path(file_path),
            filename=filename,
            document_id=document_id,
            strategy=strategy,
            language=language,
            on_progress=on_progress,
        )
```

Replace with:

```python
        result = await pipeline.run(
            file_path=Path(file_path),
            filename=filename,
            document_id=document_id,
            strategy=strategy,
            language=language,
            on_progress=on_progress,
            custom_tree=custom_tree,
        )
```

- [ ] **Step 6: Verify the module imports cleanly**

Run:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  python -c "from app.api.v1.endpoints.documents.routes import upload_document, _run_ingestion_background; print('ok')"
```

Expected output: `ok`

- [ ] **Step 7: Verify route parameter is picked up by FastAPI**

Run:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  python -c "
import inspect
from app.api.v1.endpoints.documents.routes import upload_document
sig = inspect.signature(upload_document)
assert 'page_index_file' in sig.parameters
print('ok:', list(sig.parameters))
"
```

Expected output contains `'page_index_file'`.

- [ ] **Step 8: Commit**

```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  git add app/api/v1/endpoints/documents/routes.py && \
  git commit -m "feat(upload): accept optional page_index_file and skip LLM tree when present"
```

---

### Task 7: Update OpenAPI / docstring for the upload endpoint

**Files:**
- Modify: `app/api/v1/endpoints/documents/routes.py` (docstring on `upload_document`)

The endpoint's docstring currently documents the legacy flow. A short update clarifies the new behavior for API consumers reading `/docs`.

- [ ] **Step 1: Update the docstring**

Find the existing docstring (around lines 108-113):

```python
    """Upload a PDF and start ingestion in the background.

    Returns immediately with ``status: processing``.
    Poll ``GET /documents/{document_id}`` to track progress via the
    ``status`` and ``progress_step`` fields.
    """
```

Replace with:

```python
    """Upload a PDF and start ingestion in the background.

    Returns immediately with ``status: processing``.
    Poll ``GET /documents/{document_id}`` to track progress via the
    ``status`` and ``progress_step`` fields.

    Optional ``page_index_file`` (JSON):
        When supplied, the pipeline skips LLM tree generation and uses the
        provided hierarchical tree. Schema is validated synchronously; a
        malformed or structurally-invalid file returns 400 before ingestion
        starts. Titles in each node are matched against the OCR'd markdown;
        unmatched nodes are recorded in ``ingest_warnings`` on the document
        record (see ``GET /documents/{document_id}``). Only valid with
        ``strategy='page_index'`` (or the default); rejected with 400 when
        combined with ``strategy='vector'``.
    """
```

- [ ] **Step 2: Verify module still imports**

Run:
```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  python -c "from app.api.v1.endpoints.documents.routes import upload_document; print('ok')"
```

Expected output: `ok`

- [ ] **Step 3: Commit**

```bash
cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && \
  git add app/api/v1/endpoints/documents/routes.py && \
  git commit -m "docs(upload): document page_index_file parameter on /documents/upload"
```

---

### Task 8: Manual end-to-end smoke test

This task is a manual verification checkpoint — no code changes. Run it once to confirm the full path works, then commit the plan-completion marker.

- [ ] **Step 1: Start the server in one terminal**

Run: `cd /c/Users/ACER/Desktop/compmay/dafa.ai/dafa.ai && bash run.sh`

Wait for the server to bind (look for `Uvicorn running on ...` in the logs).

- [ ] **Step 2: Prepare a minimal test JSON file**

Create a throwaway file `/tmp/test_tree.json` (or `C:\Temp\test_tree.json` on Windows) with this content:

```json
{
  "document_title": "Smoke test",
  "language": "en",
  "nodes": [
    {
      "nodeId": "1",
      "title": "Introduction",
      "summary": "Top-level intro section",
      "page_range": [1, 2],
      "children": []
    }
  ]
}
```

- [ ] **Step 3: Sanity-check validation rejects bad input**

Run (adjust the URL, port, auth if needed):

```bash
# Malformed JSON should return 400
curl -i -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer <your-token>" \
  -F "file=@/path/to/sample.pdf" \
  -F "title=smoke" \
  -F "page_index_file=@-;type=application/json;filename=bad.json" <<< '{not valid json'
```

Expected: HTTP/1.1 400 with `"page_index_file is not valid JSON"` in the body.

- [ ] **Step 4: Happy-path upload with valid tree**

Run:

```bash
curl -i -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer <your-token>" \
  -F "file=@/path/to/sample.pdf" \
  -F "title=smoke" \
  -F "page_index_file=@/tmp/test_tree.json"
```

Expected: HTTP/1.1 202 with `{"document_id": "<uuid>", "status": "processing", ...}`. Capture the `document_id`.

- [ ] **Step 5: Poll the status endpoint until ingestion completes**

Run:

```bash
curl -s http://localhost:8000/api/v1/documents/<document_id> \
  -H "Authorization: Bearer <your-token>" | python -m json.tool
```

Expected (after ingestion completes):
- `status: "completed"`
- `custom_tree_provided: true`
- `ingest_warnings: null` (or an object with `unmatched_nodes` list if the title wasn't found in the OCR'd markdown)

- [ ] **Step 6: Strategy-mismatch rejection**

If the deployment supports `strategy=vector` as a form field, verify:

```bash
curl -i -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer <your-token>" \
  -F "file=@/path/to/sample.pdf" \
  -F "title=smoke" \
  -F "strategy=vector" \
  -F "page_index_file=@/tmp/test_tree.json"
```

Expected: HTTP/1.1 400 with `"page_index_file is only valid with strategy='page_index'"`.

If the project currently has no path for the client to pass `strategy`, this case can only ever be reached by a future client change — note that and move on.

- [ ] **Step 7: Confirm LLM was skipped (log inspection)**

In the server log from Step 1, confirm for the custom-tree ingest:
- You see: `Using user-supplied page-index tree (skipping LLM generation)...`
- You do NOT see: `Building index tree: chunk 1/N ...`

- [ ] **Step 8: Optional — confirm retrieval still works**

Use whatever chat/query endpoint the project uses against the new `document_id`. Retrieval should return sensible sections. Quality degradation is expected proportional to the number of unmatched nodes (Step 5's warning count).

No commit for this task — it is verification only. If any step fails, diagnose and fix in the appropriate earlier task before re-running.

---

## Self-Review

**Spec coverage check:**

| Spec section / requirement | Task |
|---|---|
| New form field `page_index_file` | Task 6 |
| Synchronous JSON + schema validation | Task 1 (model) + Task 6 (handler) |
| Strategy-compat check (reject vector) | Task 6 |
| Thread custom_tree through pipeline | Task 5 |
| Branch around LLM loop in strategy | Task 3 |
| Recompute char offsets / ids / bboxes on custom tree | Task 3 (post-processing runs identically) |
| Warnings on unmatched titles | Task 2 (collect) + Task 3 (build dict) |
| Persist `custom_tree_provided` + `ingest_warnings` | Task 4 (repo) + Task 5 (pipeline wiring) |
| Expose on `GET /documents/{id}` | No task needed — `DocumentRepository.get()` returns the full doc including any new fields |
| OpenAPI docs | Task 7 |
| Manual verification | Task 8 |

All spec items covered.

**Placeholder scan:** No TBD / TODO / "similar to task N". Every step has concrete code or a concrete command.

**Type consistency:**
- `custom_tree: dict | None` — consistent across route, pipeline, strategy, background task.
- `ingest_warnings: dict | None` with shape `{unmatched_nodes, unmatched_count, total_nodes}` — consistent between strategy return value, pipeline → repo call, and persisted record.
- `unmatched_nodes: list[dict]` with each item `{nodeId, title}` — consistent between `_collect_unmatched` return type and strategy's warnings payload.
- Method signature `_attach_char_offsets(...) -> list[dict]` — matches Task 2 definition and Task 3 caller.

Plan is internally consistent.
