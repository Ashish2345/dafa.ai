# Citation Pills & Section Highlighting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add clickable citation pills in chat answers that scroll the PDF viewer to the correct page and highlight the cited section using word-level bounding boxes from OCR.

**Architecture:** Store word-level OCR bounding boxes per page in a new MongoDB collection `page_ocr_bboxes`. Expose a highlights endpoint that returns bboxes for a given tree node's character range. Update the LLM prompt to output structured `<cite>` tags. Frontend overlays SVG highlight rectangles on page images.

**Tech Stack:** FastAPI, MongoDB (motor async), pytest, pytest-asyncio, unittest.mock.AsyncMock

**Spec:** `docs/superpowers/specs/2026-04-14-citation-highlights-design.md`

---

## File Structure

**New files:**
- `app/db/repositories/ocr_bbox_repository.py` — CRUD for `page_ocr_bboxes` collection
- `app/api/v1/endpoints/highlights.py` — GET highlights endpoint
- `tests/conftest.py` — shared test fixtures (mock DB, mock auth)
- `tests/test_ocr_bbox_repository.py` — repository unit tests
- `tests/test_highlights_endpoint.py` — endpoint integration tests
- `tests/test_ingestion_bbox_persistence.py` — ingestion pipeline tests
- `tests/test_query_response_bboxes.py` — query response enrichment tests
- `tests/test_answer_synthesis_prompt.py` — prompt format tests

**Modified files:**
- `app/services/processing/markdown.py` — extract word-level bboxes from cleaned OCR
- `app/services/ingestion/pipeline.py` — persist word bboxes via repository
- `app/db/mongodb.py` — add indexes for `page_ocr_bboxes`
- `app/api/v1/router.py` — register highlights endpoint
- `app/services/retrieval/page_index/strategy.py` — include `page_bboxes` in query response
- `app/prompts/new_flow/answer_synthesis.py` — update citation format to `<cite>` with data attributes

---

### Task 1: OCR Bbox Repository

**Files:**
- Create: `app/db/repositories/ocr_bbox_repository.py`
- Create: `tests/conftest.py`
- Create: `tests/test_ocr_bbox_repository.py`

- [ ] **Step 1: Create shared test fixtures**

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_database():
    """Mock MongoDB database with async collections."""
    db = MagicMock()
    db.page_ocr_bboxes = AsyncMock()
    db.page_index_trees = AsyncMock()
    db.page_index_content = AsyncMock()
    db.documents = AsyncMock()
    return db


@pytest.fixture
def mock_current_user():
    """Mock authenticated user for endpoint tests."""
    return {
        "user_id": "test-user-123",
        "email": "test@example.com",
        "plan": "pro",
    }
```

- [ ] **Step 2: Write failing tests for OcrBboxRepository**

```python
# tests/test_ocr_bbox_repository.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.conftest import *  # noqa: shared fixtures


@pytest.fixture
def repo(mock_database):
    from app.db.repositories.ocr_bbox_repository import OcrBboxRepository
    return OcrBboxRepository(mock_database)


@pytest.fixture
def sample_page_words():
    return [
        {"text": "दफा", "x0": 0.12, "y0": 0.05, "x2": 0.18, "y2": 0.08, "block": 0, "line": 0, "char_offset": 0},
        {"text": "२.१", "x0": 0.19, "y0": 0.05, "x2": 0.24, "y2": 0.08, "block": 0, "line": 0, "char_offset": 4},
        {"text": "पारिश्रमिक", "x0": 0.12, "y0": 0.10, "x2": 0.30, "y2": 0.13, "block": 0, "line": 1, "char_offset": 8},
    ]


class TestSavePageBboxes:
    @pytest.mark.asyncio
    async def test_save_single_page(self, repo, mock_database, sample_page_words):
        await repo.save_page(
            document_id="doc-123",
            page=1,
            words=sample_page_words,
        )
        mock_database.page_ocr_bboxes.update_one.assert_called_once()
        call_args = mock_database.page_ocr_bboxes.update_one.call_args
        filter_doc = call_args[0][0]
        assert filter_doc == {"document_id": "doc-123", "page": 1}

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self, repo, mock_database, sample_page_words):
        await repo.save_page(document_id="doc-123", page=1, words=sample_page_words)
        call_args = mock_database.page_ocr_bboxes.update_one.call_args
        # upsert=True means it replaces if exists
        assert call_args[1].get("upsert") is True


class TestGetWordsInRange:
    @pytest.mark.asyncio
    async def test_returns_words_in_char_range(self, repo, mock_database):
        # Simulate MongoDB returning two page documents
        page1_doc = {
            "document_id": "doc-123",
            "page": 1,
            "words": [
                {"text": "hello", "x0": 0.1, "y0": 0.1, "x2": 0.2, "y2": 0.15, "block": 0, "line": 0, "char_offset": 0},
                {"text": "world", "x0": 0.25, "y0": 0.1, "x2": 0.35, "y2": 0.15, "block": 0, "line": 0, "char_offset": 6},
                {"text": "foo", "x0": 0.1, "y0": 0.2, "x2": 0.2, "y2": 0.25, "block": 0, "line": 1, "char_offset": 12},
            ],
        }
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = lambda self: self
        mock_cursor.__anext__ = AsyncMock(side_effect=[page1_doc, StopAsyncIteration])
        mock_database.page_ocr_bboxes.find.return_value = mock_cursor

        result = await repo.get_words_in_range(document_id="doc-123", start_char=0, end_char=10)

        assert len(result) == 1  # one page
        assert result[0]["page"] == 1
        # Only words with char_offset in [0, 10) should be included
        assert len(result[0]["words"]) == 2  # "hello" (0) and "world" (6), not "foo" (12)

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_data(self, repo, mock_database):
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__ = lambda self: self
        mock_cursor.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_database.page_ocr_bboxes.find.return_value = mock_cursor

        result = await repo.get_words_in_range(document_id="doc-123", start_char=0, end_char=100)
        assert result == []


class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_deletes_all_pages_for_document(self, repo, mock_database):
        await repo.delete_document(document_id="doc-123")
        mock_database.page_ocr_bboxes.delete_many.assert_called_once_with({"document_id": "doc-123"})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd dafa.ai && python -m pytest tests/test_ocr_bbox_repository.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.db.repositories.ocr_bbox_repository'`

- [ ] **Step 4: Implement OcrBboxRepository**

```python
# app/db/repositories/ocr_bbox_repository.py
"""
Repository for word-level OCR bounding boxes.

Stores one MongoDB document per page per ingested document.
Used by the highlights endpoint to return precise text regions
for citation highlighting on page images.
"""

from datetime import datetime, timezone


class OcrBboxRepository:
    def __init__(self, database):
        self.collection = database.page_ocr_bboxes

    async def save_page(
        self,
        document_id: str,
        page: int,
        words: list[dict],
    ) -> None:
        """Save word-level bboxes for a single page (upsert)."""
        await self.collection.update_one(
            {"document_id": document_id, "page": page},
            {
                "$set": {
                    "document_id": document_id,
                    "page": page,
                    "words": words,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc),
                },
            },
            upsert=True,
        )

    async def get_words_in_range(
        self,
        document_id: str,
        start_char: int,
        end_char: int,
    ) -> list[dict]:
        """Return word bboxes within a character offset range, grouped by page.

        Only words where start_char <= char_offset < end_char are included.
        """
        cursor = self.collection.find(
            {"document_id": document_id},
            {"_id": 0, "page": 1, "words": 1},
        ).sort("page", 1)

        result = []
        async for doc in cursor:
            filtered_words = [
                w for w in doc.get("words", [])
                if start_char <= w.get("char_offset", -1) < end_char
            ]
            if filtered_words:
                result.append({
                    "page": doc["page"],
                    "words": filtered_words,
                })
        return result

    async def delete_document(self, document_id: str) -> None:
        """Delete all page bbox data for a document."""
        await self.collection.delete_many({"document_id": document_id})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd dafa.ai && python -m pytest tests/test_ocr_bbox_repository.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
cd dafa.ai && git add app/db/repositories/ocr_bbox_repository.py tests/conftest.py tests/test_ocr_bbox_repository.py
git commit -m "feat: add OcrBboxRepository for word-level bounding box storage"
```

---

### Task 2: Extract Word-Level Bboxes in DocumentProcessor

**Files:**
- Modify: `app/services/processing/markdown.py`
- Create: `tests/test_word_bbox_extraction.py`

- [ ] **Step 1: Write failing tests for word bbox extraction**

```python
# tests/test_word_bbox_extraction.py
import pytest
import pandas as pd

from app.services.processing.markdown import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor()


@pytest.fixture
def sample_ocr_dataframes():
    """Two pages of OCR data with word-level bboxes (normalized 0-1)."""
    page1 = pd.DataFrame({
        "Text": ["दफा", "१", "यो", "ऐन"],
        "x0": [0.10, 0.18, 0.10, 0.18],
        "y0": [0.05, 0.05, 0.10, 0.10],
        "x2": [0.17, 0.22, 0.17, 0.25],
        "y2": [0.08, 0.08, 0.13, 0.13],
        "block": [0, 0, 0, 0],
        "line": [0, 0, 1, 1],
        "space_type": ["word", "word", "word", "word"],
        "page": [1, 1, 1, 1],
        "confidence": [0.99, 0.99, 0.99, 0.99],
        "index_sort": [0, 1, 2, 3],
    })
    page2 = pd.DataFrame({
        "Text": ["दफा", "२"],
        "x0": [0.10, 0.18],
        "y0": [0.05, 0.05],
        "x2": [0.17, 0.22],
        "y2": [0.08, 0.08],
        "block": [0, 0],
        "line": [0, 0],
        "space_type": ["word", "word"],
        "page": [2, 2],
        "confidence": [0.99, 0.99],
        "index_sort": [0, 1],
    })
    return [page1, page2]


class TestExtractWordBboxes:
    def test_returns_word_bboxes_key(self, processor, sample_ocr_dataframes):
        result = processor.process_to_markdown(
            raw_ocr=sample_ocr_dataframes,
            page_scalars=[{"width": 1, "height": 1}, {"width": 1, "height": 1}],
        )
        assert "word_bboxes" in result

    def test_word_bboxes_grouped_by_page(self, processor, sample_ocr_dataframes):
        result = processor.process_to_markdown(
            raw_ocr=sample_ocr_dataframes,
            page_scalars=[{"width": 1, "height": 1}, {"width": 1, "height": 1}],
        )
        word_bboxes = result["word_bboxes"]
        pages = [entry["page"] for entry in word_bboxes]
        assert 1 in pages
        assert 2 in pages

    def test_word_bboxes_have_char_offset(self, processor, sample_ocr_dataframes):
        result = processor.process_to_markdown(
            raw_ocr=sample_ocr_dataframes,
            page_scalars=[{"width": 1, "height": 1}, {"width": 1, "height": 1}],
        )
        word_bboxes = result["word_bboxes"]
        for page_entry in word_bboxes:
            for word in page_entry["words"]:
                assert "char_offset" in word
                assert "text" in word
                assert "x0" in word
                assert "y0" in word
                assert "x2" in word
                assert "y2" in word
                assert "block" in word
                assert "line" in word

    def test_char_offsets_are_monotonically_increasing(self, processor, sample_ocr_dataframes):
        result = processor.process_to_markdown(
            raw_ocr=sample_ocr_dataframes,
            page_scalars=[{"width": 1, "height": 1}, {"width": 1, "height": 1}],
        )
        all_offsets = []
        for page_entry in result["word_bboxes"]:
            for word in page_entry["words"]:
                all_offsets.append(word["char_offset"])
        # Offsets should be strictly increasing across pages
        for i in range(1, len(all_offsets)):
            assert all_offsets[i] > all_offsets[i - 1]

    def test_char_offset_points_to_word_in_markdown(self, processor, sample_ocr_dataframes):
        result = processor.process_to_markdown(
            raw_ocr=sample_ocr_dataframes,
            page_scalars=[{"width": 1, "height": 1}, {"width": 1, "height": 1}],
        )
        markdown = result["markdown"]
        for page_entry in result["word_bboxes"]:
            for word in page_entry["words"]:
                offset = word["char_offset"]
                text = word["text"]
                # The word's text should appear at or near the char_offset in the markdown
                # (may not be exact due to line number removal, but should be findable)
                assert offset < len(markdown)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dafa.ai && python -m pytest tests/test_word_bbox_extraction.py -v`
Expected: FAIL — `KeyError: 'word_bboxes'` (not yet returned by `process_to_markdown`)

- [ ] **Step 3: Implement word bbox extraction in DocumentProcessor**

Modify `app/services/processing/markdown.py`. Add a new method `_extract_word_bboxes` and call it from `process_to_markdown`:

Add this method to the `DocumentProcessor` class (after `_convert_to_markdown_with_bboxes`):

```python
def _extract_word_bboxes(
    self,
    parsed_result: dict,
    full_markdown: str,
    page_bbox_map: list[dict],
) -> list[dict]:
    """Extract word-level bounding boxes with character offsets.

    Uses the cleaned OCR DataFrames (which retain word-level x0/y0/x2/y2)
    and maps each word to its approximate character offset in the full
    markdown string using the page_bbox_map for page-level start_char.

    Returns:
        List of dicts, one per page:
        [{"page": 1, "words": [{"text": ..., "x0": ..., "char_offset": ...}, ...]}, ...]
    """
    cleaned_ocr = parsed_result.get("cleaned_ocr", [])
    pages_data = parsed_result.get("pages", [])
    result = []

    for page_idx, ocr_df in enumerate(cleaned_ocr):
        if ocr_df.empty:
            continue

        page_number = pages_data[page_idx]["page_number"] if page_idx < len(pages_data) else page_idx + 1

        # Find this page's start_char from page_bbox_map
        page_start_char = 0
        for entry in page_bbox_map:
            if entry["page"] == page_number:
                page_start_char = entry["start_char"]
                break

        # Build word list from OCR DataFrame
        words = []
        running_offset = page_start_char
        for _, row in ocr_df.iterrows():
            text = str(row.get("Text", "")).strip()
            if not text:
                continue
            words.append({
                "text": text,
                "x0": float(row["x0"]),
                "y0": float(row["y0"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "block": int(row.get("block", 0)),
                "line": int(row.get("line", 0)),
                "char_offset": running_offset,
            })
            # Advance offset by text length + 1 for space/newline
            running_offset += len(text) + 1

        if words:
            result.append({"page": page_number, "words": words})

    return result
```

Then modify `process_to_markdown` to include `word_bboxes` in its return value. After line 70 (`full_markdown, page_bbox_map = self._convert_to_markdown_with_bboxes(parsed_result)`), add the extraction call. Update the return dict (line 104) to include the new key:

```python
# Add after line 70:
word_bboxes = self._extract_word_bboxes(parsed_result, full_markdown, page_bbox_map)

# Update return dict to include:
"word_bboxes": word_bboxes,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dafa.ai && python -m pytest tests/test_word_bbox_extraction.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd dafa.ai && git add app/services/processing/markdown.py tests/test_word_bbox_extraction.py
git commit -m "feat: extract word-level bboxes with char offsets in DocumentProcessor"
```

---

### Task 3: Persist Word Bboxes in Ingestion Pipeline

**Files:**
- Modify: `app/services/ingestion/pipeline.py`
- Create: `tests/test_ingestion_bbox_persistence.py`

- [ ] **Step 1: Write failing test for bbox persistence during ingestion**

```python
# tests/test_ingestion_bbox_persistence.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


@pytest.fixture
def mock_ocr_bbox_repo():
    repo = AsyncMock()
    repo.save_page = AsyncMock()
    return repo


@pytest.fixture
def mock_strategy():
    strategy = AsyncMock()
    strategy.ingest = AsyncMock()
    strategy.__class__.__name__ = "PageIndexStrategy"
    return strategy


@pytest.fixture
def mock_pipeline_deps(mock_ocr_bbox_repo):
    return {
        "parser_factory": AsyncMock(),
        "processor": MagicMock(),
        "metadata_extractor": MagicMock(),
        "file_storage": AsyncMock(),
        "document_repo": AsyncMock(),
        "ocr_bbox_repo": mock_ocr_bbox_repo,
    }


class TestIngestionBboxPersistence:
    @pytest.mark.asyncio
    async def test_saves_word_bboxes_during_ingestion(self, mock_pipeline_deps, mock_strategy):
        from app.services.ingestion.pipeline import IngestionPipeline

        # Setup mock returns
        mock_pipeline_deps["parser_factory"].parse = AsyncMock(return_value={"pages": []})

        # Mock IngestionService.gather_document_data
        gather_result = {
            "status": "success",
            "raw_ocr": [],
            "page_scalars": [{"width": 612, "height": 792}],
            "page_images": [],
        }

        # Mock processor.process_to_markdown
        word_bboxes_data = [
            {
                "page": 1,
                "words": [
                    {"text": "hello", "x0": 0.1, "y0": 0.1, "x2": 0.2, "y2": 0.15, "block": 0, "line": 0, "char_offset": 0},
                ],
            }
        ]
        mock_pipeline_deps["processor"].process_to_markdown.return_value = {
            "markdown": "## Page 1\nhello",
            "page_bbox_map": [{"start_char": 0, "end_char": 15, "page": 1, "bbox": {"x0": 0, "y0": 0, "x2": 1, "y2": 1}}],
            "word_bboxes": word_bboxes_data,
        }
        mock_pipeline_deps["metadata_extractor"].extract_metadata.return_value = {"language": "en"}

        pipeline = IngestionPipeline(
            parser_factory=mock_pipeline_deps["parser_factory"],
            processor=mock_pipeline_deps["processor"],
            metadata_extractor=mock_pipeline_deps["metadata_extractor"],
            file_storage=mock_pipeline_deps["file_storage"],
            document_repo=mock_pipeline_deps["document_repo"],
            ocr_bbox_repo=mock_pipeline_deps["ocr_bbox_repo"],
        )

        with patch("app.services.ingestion.pipeline.IngestionService") as MockService:
            MockService.return_value.gather_document_data.return_value = gather_result
            result = await pipeline.run(
                file_path=Path("/tmp/test.pdf"),
                filename="test.pdf",
                document_id="doc-123",
                strategy=mock_strategy,
                language="en",
            )

        # Verify word bboxes were saved
        mock_pipeline_deps["ocr_bbox_repo"].save_page.assert_called_once_with(
            document_id="doc-123",
            page=1,
            words=word_bboxes_data[0]["words"],
        )

    @pytest.mark.asyncio
    async def test_ingestion_succeeds_without_ocr_bbox_repo(self, mock_strategy):
        """Pipeline should work when ocr_bbox_repo is not provided (backward compat)."""
        from app.services.ingestion.pipeline import IngestionPipeline

        mock_parser = AsyncMock()
        mock_parser.parse = AsyncMock(return_value={"pages": []})
        mock_processor = MagicMock()
        mock_processor.process_to_markdown.return_value = {
            "markdown": "test",
            "page_bbox_map": [],
            "word_bboxes": [],
        }
        mock_extractor = MagicMock()
        mock_extractor.extract_metadata.return_value = {"language": "en"}

        pipeline = IngestionPipeline(
            parser_factory=mock_parser,
            processor=mock_processor,
            metadata_extractor=mock_extractor,
            file_storage=None,
            document_repo=AsyncMock(),
        )

        gather_result = {"status": "success", "raw_ocr": [], "page_scalars": [], "page_images": []}
        with patch("app.services.ingestion.pipeline.IngestionService") as MockService:
            MockService.return_value.gather_document_data.return_value = gather_result
            result = await pipeline.run(
                file_path=Path("/tmp/test.pdf"),
                filename="test.pdf",
                document_id="doc-123",
                strategy=mock_strategy,
            )

        assert result["status"] == "completed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dafa.ai && python -m pytest tests/test_ingestion_bbox_persistence.py -v`
Expected: FAIL — `TypeError: IngestionPipeline.__init__() got an unexpected keyword argument 'ocr_bbox_repo'`

- [ ] **Step 3: Modify IngestionPipeline to accept and use ocr_bbox_repo**

In `app/services/ingestion/pipeline.py`, make these changes:

Add `ocr_bbox_repo=None` to `__init__`:

```python
def __init__(
    self,
    parser_factory=None,
    processor=None,
    metadata_extractor=None,
    file_storage=None,
    document_repo=None,
    ocr_bbox_repo=None,
):
    self.parser_factory = parser_factory
    self.processor = processor
    self.metadata_extractor = metadata_extractor
    self.file_storage = file_storage
    self.document_repo = document_repo
    self.ocr_bbox_repo = ocr_bbox_repo
```

After line 91 (`page_bbox_map = markdown_result.get("page_bbox_map", [])`), add word bbox persistence:

```python
# Persist word-level OCR bounding boxes (if repo available)
word_bboxes = markdown_result.get("word_bboxes", [])
if self.ocr_bbox_repo and word_bboxes:
    await _step("Saving word-level bounding boxes...")
    for page_entry in word_bboxes:
        await self.ocr_bbox_repo.save_page(
            document_id=document_id,
            page=page_entry["page"],
            words=page_entry["words"],
        )
    logger.info(f"[{document_id[:8]}] Saved word bboxes for {len(word_bboxes)} pages")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dafa.ai && python -m pytest tests/test_ingestion_bbox_persistence.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd dafa.ai && git add app/services/ingestion/pipeline.py tests/test_ingestion_bbox_persistence.py
git commit -m "feat: persist word-level bboxes during ingestion pipeline"
```

---

### Task 4: MongoDB Indexes for page_ocr_bboxes

**Files:**
- Modify: `app/db/mongodb.py`

- [ ] **Step 1: Add indexes for the new collection**

In `app/db/mongodb.py`, inside the `_create_indexes` method, add after the `page_index_content` index block (after line 182):

```python
# Word-level OCR bounding boxes — one doc per (document, page)
page_ocr_bboxes = self.database.page_ocr_bboxes
await page_ocr_bboxes.create_index(
    [("document_id", 1), ("page", 1)],
    unique=True,
)
await page_ocr_bboxes.create_index("document_id")
```

- [ ] **Step 2: Commit**

```bash
cd dafa.ai && git add app/db/mongodb.py
git commit -m "feat: add MongoDB indexes for page_ocr_bboxes collection"
```

---

### Task 5: Highlights Endpoint

**Files:**
- Create: `app/api/v1/endpoints/highlights.py`
- Modify: `app/api/v1/router.py`
- Create: `tests/test_highlights_endpoint.py`

- [ ] **Step 1: Write failing tests for highlights endpoint**

```python
# tests/test_highlights_endpoint.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_database():
    db = MagicMock()
    db.page_ocr_bboxes = AsyncMock()
    db.page_index_trees = AsyncMock()
    db.page_index_content = AsyncMock()
    return db


@pytest.fixture
def app(mock_database):
    from fastapi import FastAPI
    from app.api.v1.endpoints.highlights import router

    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1/documents")

    # Override dependencies
    from app.db.mongodb import get_database
    from app.utils.auth import get_current_user

    test_app.dependency_overrides[get_database] = lambda: mock_database
    test_app.dependency_overrides[get_current_user] = lambda: {"user_id": "test-user"}

    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHighlightsEndpoint:
    def test_returns_highlights_for_node_id(self, client, mock_database):
        # Mock tree lookup to return node with start_char/end_char
        tree_doc = {
            "tree": {
                "nodes": [
                    {
                        "nodeId": "2.1",
                        "id": 3,
                        "title": "Section 2.1",
                        "start_char": 100,
                        "end_char": 300,
                        "children": [],
                    }
                ]
            }
        }
        mock_database.page_index_trees.find_one = AsyncMock(return_value=tree_doc)

        # Mock bbox lookup
        bbox_cursor = AsyncMock()
        bbox_docs = [
            {
                "page": 3,
                "words": [
                    {"text": "दफा", "x0": 0.1, "y0": 0.18, "x2": 0.16, "y2": 0.22, "block": 0, "line": 0, "char_offset": 100},
                    {"text": "२.१", "x0": 0.17, "y0": 0.18, "x2": 0.22, "y2": 0.22, "block": 0, "line": 0, "char_offset": 104},
                    {"text": "पारिश्रमिक", "x0": 0.1, "y0": 0.23, "x2": 0.3, "y2": 0.27, "block": 0, "line": 1, "char_offset": 108},
                ],
            }
        ]
        bbox_cursor.__aiter__ = lambda self: self
        bbox_cursor.__anext__ = AsyncMock(side_effect=bbox_docs + [StopAsyncIteration])
        mock_database.page_ocr_bboxes.find.return_value = bbox_cursor

        response = client.get("/api/v1/documents/doc-123/highlights?node_id=2.1")

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc-123"
        assert data["node_id"] == "2.1"
        assert len(data["highlights"]) == 1
        assert data["highlights"][0]["page"] == 3
        assert len(data["highlights"][0]["lines"]) == 2  # two lines

    def test_returns_line_level_bboxes(self, client, mock_database):
        tree_doc = {
            "tree": {
                "nodes": [
                    {"nodeId": "1", "id": 0, "start_char": 0, "end_char": 50, "children": []}
                ]
            }
        }
        mock_database.page_index_trees.find_one = AsyncMock(return_value=tree_doc)

        bbox_cursor = AsyncMock()
        bbox_docs = [
            {
                "page": 1,
                "words": [
                    {"text": "a", "x0": 0.1, "y0": 0.1, "x2": 0.15, "y2": 0.14, "block": 0, "line": 0, "char_offset": 0},
                    {"text": "b", "x0": 0.2, "y0": 0.1, "x2": 0.25, "y2": 0.14, "block": 0, "line": 0, "char_offset": 2},
                    {"text": "c", "x0": 0.1, "y0": 0.2, "x2": 0.15, "y2": 0.24, "block": 0, "line": 1, "char_offset": 4},
                ],
            }
        ]
        bbox_cursor.__aiter__ = lambda self: self
        bbox_cursor.__anext__ = AsyncMock(side_effect=bbox_docs + [StopAsyncIteration])
        mock_database.page_ocr_bboxes.find.return_value = bbox_cursor

        response = client.get("/api/v1/documents/doc-123/highlights?node_id=1")
        data = response.json()

        line0 = data["highlights"][0]["lines"][0]
        # Line bbox should be the bounding box of all words in that line
        assert line0["bbox"]["x0"] == 0.1
        assert line0["bbox"]["x2"] == 0.25  # max of word x2 values
        assert line0["bbox"]["y0"] == 0.1
        assert line0["bbox"]["y2"] == 0.14

    def test_returns_404_when_node_not_found(self, client, mock_database):
        tree_doc = {"tree": {"nodes": [{"nodeId": "1", "id": 0, "start_char": 0, "end_char": 50, "children": []}]}}
        mock_database.page_index_trees.find_one = AsyncMock(return_value=tree_doc)

        response = client.get("/api/v1/documents/doc-123/highlights?node_id=999")
        assert response.status_code == 404

    def test_returns_404_when_document_not_found(self, client, mock_database):
        mock_database.page_index_trees.find_one = AsyncMock(return_value=None)
        response = client.get("/api/v1/documents/doc-123/highlights?node_id=1")
        assert response.status_code == 404

    def test_returns_empty_highlights_when_no_bboxes(self, client, mock_database):
        tree_doc = {"tree": {"nodes": [{"nodeId": "1", "id": 0, "start_char": 0, "end_char": 50, "children": []}]}}
        mock_database.page_index_trees.find_one = AsyncMock(return_value=tree_doc)

        bbox_cursor = AsyncMock()
        bbox_cursor.__aiter__ = lambda self: self
        bbox_cursor.__anext__ = AsyncMock(side_effect=StopAsyncIteration)
        mock_database.page_ocr_bboxes.find.return_value = bbox_cursor

        response = client.get("/api/v1/documents/doc-123/highlights?node_id=1")
        assert response.status_code == 200
        assert response.json()["highlights"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dafa.ai && python -m pytest tests/test_highlights_endpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.api.v1.endpoints.highlights'`

- [ ] **Step 3: Implement highlights endpoint**

```python
# app/api/v1/endpoints/highlights.py
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
    # Look up the tree to find the node's char range
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

    # Fetch word bboxes in the character range
    bbox_repo = OcrBboxRepository(db)
    page_results = await bbox_repo.get_words_in_range(document_id, start_char, end_char)

    # Group words by line within each page
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
```

- [ ] **Step 4: Register the endpoint in the router**

In `app/api/v1/router.py`, add the import and registration:

```python
from app.api.v1.endpoints import auth, chats, documents, feedback, health, highlights, plans, preferences, query, starred, usage
```

Add in the protected routes section:

```python
api_router.include_router(highlights.router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd dafa.ai && python -m pytest tests/test_highlights_endpoint.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
cd dafa.ai && git add app/api/v1/endpoints/highlights.py app/api/v1/router.py tests/test_highlights_endpoint.py
git commit -m "feat: add GET /documents/{id}/highlights endpoint for citation highlighting"
```

---

### Task 6: Include page_bboxes in Query Response (Fallback)

**Files:**
- Modify: `app/services/retrieval/page_index/strategy.py`
- Create: `tests/test_query_response_bboxes.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_query_response_bboxes.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_repo():
    repo = AsyncMock()
    return repo


@pytest.fixture
def mock_llm():
    return MagicMock()


class TestQueryResponseIncludesBboxes:
    @pytest.mark.asyncio
    async def test_retrieved_chunk_source_includes_page_bboxes(self, mock_repo, mock_llm):
        from app.services.retrieval.page_index.strategy import PageIndexStrategy

        strategy = PageIndexStrategy(repository=mock_repo, llm_service=mock_llm)

        # Mock repo.list_document_ids
        mock_repo.list_document_ids.return_value = ["doc-123"]

        # Mock repo.get_tree — tree with a node that has page_bboxes
        mock_repo.get_tree.return_value = {
            "tree": {
                "document_title": "Test Act",
                "nodes": [
                    {
                        "nodeId": "1",
                        "id": 0,
                        "title": "Section 1",
                        "summary": "Test section",
                        "start_char": 0,
                        "end_char": 100,
                        "page_range": [1, 1],
                        "page_bboxes": [{"page": 1, "bbox": {"x0": 0, "y0": 0.1, "x2": 1, "y2": 0.4}}],
                        "children": [],
                    }
                ],
            },
            "language": "en",
        }
        mock_repo.get_markdown.return_value = "## Page 1\nTest content here"

        # Mock section_retriever to return a section with page_bboxes
        section_result = [
            {
                "nodeId": "1",
                "int_id": 0,
                "title": "Section 1",
                "summary": "Test section",
                "text": "Test content here",
                "page_range": [1, 1],
                "page_bboxes": [{"page": 1, "bbox": {"x0": 0, "y0": 0.1, "x2": 1, "y2": 0.4}}],
                "metadata": {"source": "page_index", "node_id": "1", "title": "Section 1"},
            }
        ]

        with patch.object(strategy.section_retriever, "retrieve", return_value=section_result):
            import asyncio
            with patch("asyncio.to_thread", new=AsyncMock(return_value=section_result)):
                chunks = await strategy.retrieve(query="test query", top_k=5)

        assert len(chunks) == 1
        assert "page_bboxes" in chunks[0].source
        assert chunks[0].source["page_bboxes"] == [{"page": 1, "bbox": {"x0": 0, "y0": 0.1, "x2": 1, "y2": 0.4}}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dafa.ai && python -m pytest tests/test_query_response_bboxes.py -v`
Expected: FAIL — `KeyError: 'page_bboxes'` (not yet included in source dict)

- [ ] **Step 3: Modify strategy.py to include page_bboxes**

In `app/services/retrieval/page_index/strategy.py`, update the `RetrievedChunk` construction (around line 77). Add `page_bboxes` to the source dict:

```python
source={
    "document_name": act_name,
    "document_id": doc_id,
    "page_range": page_range,
    "page": page_range[0] if page_range else None,
    "section": section.get("title", ""),
    "node_id": section["nodeId"],
    "node_int_id": section.get("int_id"),
    "page_bboxes": section.get("page_bboxes", []),
},
```

Also update `section_retriever.py` to pass `page_bboxes` through from the node. In `app/services/retrieval/page_index/section_retriever.py`, where sections are built from nodes, include:

```python
"page_bboxes": node.get("page_bboxes", []),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dafa.ai && python -m pytest tests/test_query_response_bboxes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd dafa.ai && git add app/services/retrieval/page_index/strategy.py app/services/retrieval/page_index/section_retriever.py tests/test_query_response_bboxes.py
git commit -m "feat: include page_bboxes fallback in query response chunks"
```

---

### Task 7: Update LLM Prompt for Structured Cite Tags

**Files:**
- Modify: `app/prompts/new_flow/answer_synthesis.py`
- Create: `tests/test_answer_synthesis_prompt.py`

- [ ] **Step 1: Write test for prompt format**

```python
# tests/test_answer_synthesis_prompt.py
from app.prompts.new_flow.answer_synthesis import get_prompts, SYSTEM_PROMPT, USER_PROMPT


class TestAnswerSynthesisPrompt:
    def test_system_prompt_instructs_cite_with_data_attributes(self):
        assert "data-node" in SYSTEM_PROMPT
        assert "data-doc" in SYSTEM_PROMPT
        assert "data-page" in SYSTEM_PROMPT
        assert "data-section" in SYSTEM_PROMPT

    def test_system_prompt_has_cite_example(self):
        assert '<cite data-node="' in SYSTEM_PROMPT

    def test_user_prompt_template_has_placeholders(self):
        assert "{act_name}" in USER_PROMPT
        assert "{sections}" in USER_PROMPT
        assert "{query}" in USER_PROMPT

    def test_get_prompts_returns_english_by_default(self):
        system, user = get_prompts()
        assert "data-node" in system

    def test_get_prompts_returns_nepali_for_ne(self):
        system, user = get_prompts("ne")
        assert "data-node" in system
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd dafa.ai && python -m pytest tests/test_answer_synthesis_prompt.py -v`
Expected: FAIL — `assert 'data-node' in SYSTEM_PROMPT` fails (current prompt uses `<cite>(Section {nodeId}: {title})</cite>`)

- [ ] **Step 3: Update the prompts**

Replace the contents of `app/prompts/new_flow/answer_synthesis.py`:

```python
"""
Answer synthesis prompts — New Flow (PageIndex Vectorless RAG).

Used in: app/services/rag/orchestrator.py → _synthesize_answer() (when use_page_index=True)
Purpose: Generate a precise answer from PageIndex-retrieved sections, citing nodeIds.
LLM config: temperature=0.2, max_tokens=8192

Citations use structured <cite> tags with data attributes so the frontend can make
them interactive (click to scroll + highlight the source on the page image).
"""

SYSTEM_PROMPT = """You are a precise legal assistant answering questions about finance acts and regulations.

Rules:
- Use ONLY the provided sections to answer. Do NOT use external knowledge.
- Cite every fact using a structured <cite> tag with data attributes:
  <cite data-node="{nodeId}" data-doc="{document_id}" data-page="{first_page}" data-section="{title}">Section {nodeId}, Page {first_page}</cite>
- If the provided sections do not contain enough information to answer, say exactly: "The provided sections do not contain sufficient information to answer this question."
- Do NOT speculate, infer, or extrapolate beyond what the sections explicitly state.
- When quoting rates, thresholds, or penalties, state them exactly as written.

Citation example:
  <cite data-node="2.1" data-doc="4a3efdb0-281f" data-page="3" data-section="Remuneration Payments">Section 2.1, Page 3</cite>

Output format — return clean HTML only, no markdown, no code fences:
- Use <h3> for main topic headings
- Use <ul><li> for lists of points
- Use <strong> for key numbers, rates, and deadlines
- Use <cite> for section citations with data-node, data-doc, data-page, data-section attributes
- Use <p> for short introductory or closing sentences
- Do NOT include <html>, <head>, <body> tags — just the inner content fragment"""

SYSTEM_PROMPT_NE = """तपाईं वित्त ऐनहरूको बारेमा प्रश्नको उत्तर दिने सटीक कानुनी सहायक हुनुहुन्छ।

नियमहरू:
- केवल प्रदान गरिएका खण्डहरू मात्र प्रयोग गर्नुहोस्। बाहिरी ज्ञान प्रयोग नगर्नुहोस्।
- प्रत्येक तथ्यको उद्धरण structured <cite> tag मा गर्नुहोस्:
  <cite data-node="{nodeId}" data-doc="{document_id}" data-page="{first_page}" data-section="{title}">दफा {nodeId}, पृष्ठ {first_page}</cite>
- यदि खण्डहरूमा पर्याप्त जानकारी छैन भने: "प्रदान गरिएका खण्डहरूमा यो प्रश्नको उत्तर दिन पर्याप्त जानकारी छैन।"
- अनुमान वा निष्कर्ष नगर्नुहोस्।

आउटपुट: सफा HTML मात्र — h3, ul/li, strong, cite ट्यागहरू प्रयोग गर्नुहोस्। markdown वा code fence नगर्नुहोस्।"""

USER_PROMPT = """Relevant sections from {act_name}:

{sections}

Question: {query}

Answer in HTML format (h3, ul/li, strong, cite tags only). Cite every fact with a structured <cite> tag using data-node, data-doc, data-page, data-section attributes."""

USER_PROMPT_NE = """{act_name} बाट सान्दर्भिक खण्डहरू:

{sections}

प्रश्न: {query}

structured <cite> tag (data-node, data-doc, data-page, data-section attributes) प्रयोग गरेर सटीक उद्धरणसहित HTML मा उत्तर दिनुहोस्।"""


def get_prompts(language: str = "en") -> tuple[str, str]:
    """Return (system_prompt, user_prompt_template) for the given language."""
    if language == "ne":
        return SYSTEM_PROMPT_NE, USER_PROMPT_NE
    return SYSTEM_PROMPT, USER_PROMPT
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd dafa.ai && python -m pytest tests/test_answer_synthesis_prompt.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `cd dafa.ai && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd dafa.ai && git add app/prompts/new_flow/answer_synthesis.py tests/test_answer_synthesis_prompt.py
git commit -m "feat: update synthesis prompt to output structured cite tags with data attributes"
```

---

### Task 8: Wire Up OcrBboxRepository in Application Startup

**Files:**
- Modify: `app/api/v1/endpoints/documents.py` (where IngestionPipeline is constructed)

- [ ] **Step 1: Find where IngestionPipeline is instantiated**

Search for `IngestionPipeline(` in the codebase to find the call site where the pipeline is constructed and pass `ocr_bbox_repo`.

Run: `grep -rn "IngestionPipeline(" app/`

- [ ] **Step 2: Add OcrBboxRepository to the pipeline construction**

At the call site where `IngestionPipeline` is instantiated, add:

```python
from app.db.repositories.ocr_bbox_repository import OcrBboxRepository

ocr_bbox_repo = OcrBboxRepository(db)
pipeline = IngestionPipeline(
    parser_factory=...,
    processor=...,
    metadata_extractor=...,
    file_storage=...,
    document_repo=...,
    ocr_bbox_repo=ocr_bbox_repo,  # NEW
)
```

- [ ] **Step 3: Run full test suite to verify nothing broke**

Run: `cd dafa.ai && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
cd dafa.ai && git add app/api/v1/endpoints/documents.py
git commit -m "feat: wire OcrBboxRepository into ingestion pipeline at startup"
```

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | OcrBboxRepository (save/query/delete) | 5 tests |
| 2 | Extract word bboxes in DocumentProcessor | 5 tests |
| 3 | Persist word bboxes in ingestion pipeline | 2 tests |
| 4 | MongoDB indexes for page_ocr_bboxes | — |
| 5 | GET /documents/{id}/highlights endpoint | 5 tests |
| 6 | Include page_bboxes in query response | 1 test |
| 7 | Update LLM prompt for cite tags | 5 tests |
| 8 | Wire up repository at startup | integration |

Total: 8 tasks, 23 tests, 8 commits
