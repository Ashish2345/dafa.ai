"""
Abstract base classes for PDF reader and backends.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from .page import PDFPage


class PDFBackend(ABC):
    """
    Abstract base class for PDF rendering backends.

    Backends are responsible for loading PDFs and rendering individual pages
    to numpy arrays.
    """

    @abstractmethod
    def render_page(self, index: int, dpi: int) -> np.ndarray:
        """
        Render a single page to a numpy array.

        Args:
            index: Zero-based page index
            dpi: Dots per inch for rendering

        Returns:
            numpy array in BGR format (OpenCV compatible)
        """
        pass

    @property
    @abstractmethod
    def page_count(self) -> int:
        """Return the total number of pages in the PDF."""
        pass

    @abstractmethod
    def get_page_dimensions(self, index: int) -> tuple[int, int]:
        """
        Get the dimensions of a page in points (1/72 inch).

        Args:
            index: Zero-based page index

        Returns:
            Tuple of (width, height) in points
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend."""
        pass

    @abstractmethod
    def get_page_words(self, index: int) -> list:
        """
        Get words from a page.

        Args:
            index: Zero-based page index

        Returns:
            List of tuples: (x0, y0, x2, y2, text, block_no, line_no, word_no)
            Returns empty list if word extraction is not supported.
        """
        pass


class PDFReaderBase(ABC):
    """
    Abstract base class for PDF readers.

    Provides the interface for iterating over PDF pages lazily.
    Supports context manager protocol for resource management.
    """

    @property
    @abstractmethod
    def path(self) -> str:
        """Return the path to the PDF file."""
        pass

    @property
    @abstractmethod
    def page_count(self) -> int:
        """Return the total number of pages."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator["PDFPage"]:
        """Iterate over all pages in the PDF."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of pages."""
        pass

    @abstractmethod
    def __enter__(self) -> "PDFReaderBase":
        """Enter context manager."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and release resources."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the reader."""
        pass
