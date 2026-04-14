import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.conftest import *  # noqa: shared fixtures


class AsyncCursor:
    """Minimal async cursor helper for mocking motor's cursor interface."""

    def __init__(self, docs):
        self._iter = iter(docs)

    def sort(self, *args, **kwargs):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


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
        assert call_args[1].get("upsert") is True


class TestGetWordsInRange:
    @pytest.mark.asyncio
    async def test_returns_words_in_char_range(self, repo, mock_database):
        page1_doc = {
            "document_id": "doc-123",
            "page": 1,
            "words": [
                {"text": "hello", "x0": 0.1, "y0": 0.1, "x2": 0.2, "y2": 0.15, "block": 0, "line": 0, "char_offset": 0},
                {"text": "world", "x0": 0.25, "y0": 0.1, "x2": 0.35, "y2": 0.15, "block": 0, "line": 0, "char_offset": 6},
                {"text": "foo", "x0": 0.1, "y0": 0.2, "x2": 0.2, "y2": 0.25, "block": 0, "line": 1, "char_offset": 12},
            ],
        }
        mock_cursor = AsyncCursor([page1_doc])
        mock_database.page_ocr_bboxes.find = MagicMock(return_value=mock_cursor)

        result = await repo.get_words_in_range(document_id="doc-123", start_char=0, end_char=10)

        assert len(result) == 1
        assert result[0]["page"] == 1
        assert len(result[0]["words"]) == 2  # "hello" (0) and "world" (6), not "foo" (12)

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_data(self, repo, mock_database):
        mock_cursor = AsyncCursor([])
        mock_database.page_ocr_bboxes.find = MagicMock(return_value=mock_cursor)

        result = await repo.get_words_in_range(document_id="doc-123", start_char=0, end_char=100)
        assert result == []


class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_deletes_all_pages_for_document(self, repo, mock_database):
        await repo.delete_document(document_id="doc-123")
        mock_database.page_ocr_bboxes.delete_many.assert_called_once_with({"document_id": "doc-123"})
