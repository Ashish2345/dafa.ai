"""
Retrieval strategy interface.

All retrieval approaches (PageIndex, Vector) implement this ABC.
The factory selects the right implementation per-request.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional


@dataclass
class RetrievedChunk:
    """Common output shape for all retrieval strategies."""

    text: str
    source: dict[str, Any]
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalStrategy(ABC):
    """
    Abstract base for retrieval strategies.

    Each strategy implements both retrieval and ingestion because
    storage format is strategy-specific (trees vs embeddings).
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return ranked chunks relevant to the query."""
        ...

    @abstractmethod
    async def ingest(
        self,
        document_id: str,
        markdown: str,
        metadata: dict[str, Any],
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
        page_bbox_map: list[dict] | None = None,
        image_dimensions: dict | None = None,
    ) -> None:
        """Store a processed document for later retrieval."""
        ...
