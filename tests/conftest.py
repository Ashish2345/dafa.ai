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
