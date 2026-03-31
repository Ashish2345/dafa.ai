"""
Base types and abstract document source for batch injection.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentInfo:
    """A single discoverable document to fetch and ingest."""

    title: str
    category: str  # acts_rules, finance_acts, nrb, ird, gazette, local
    source_id: str
    url: Optional[str] = None
    local_path: Optional[str] = None
    year: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_source_id(category: str, url: Optional[str], local_path: Optional[str], title: str) -> str:
        """Deterministic id for deduplication in injection_log."""
        key = f"{category}|{url or ''}|{local_path or ''}|{title}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class BaseDocumentSource(ABC):
    """Source of PDFs (or other ingestible files) for a config block under sources.yaml."""

    def __init__(self, category_key: str, config: Dict[str, Any], scraper_delay_s: float = 1.0) -> None:
        self.category_key = category_key
        self.config = config
        self.scraper_delay_s = scraper_delay_s

    @abstractmethod
    async def discover(
        self,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[DocumentInfo]:
        """Discover available documents, optionally filtered by BS year range (inclusive)."""

    @abstractmethod
    async def fetch(self, doc_info: DocumentInfo) -> str:
        """Download or copy the document and return a local filesystem path."""


def year_in_range(year: Optional[int], year_from: Optional[int], year_to: Optional[int]) -> bool:
    """
    True if the document year is acceptable for the requested range.

    If neither year_from nor year_to is set, all documents match (including unknown year).
    If either bound is set, documents without a parseable year are excluded.
    """
    if year_from is None and year_to is None:
        return True
    if year is None:
        return False
    if year_from is not None and year < year_from:
        return False
    if year_to is not None and year > year_to:
        return False
    return True
