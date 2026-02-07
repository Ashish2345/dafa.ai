"""
PDFReader - Main PDF reader class with lazy page iteration.
"""

from typing import Iterator, Optional, Tuple

from loguru import logger

from .backends import FitzBackend, PdftoppmBackend
from .base import PDFBackend, PDFReaderBase
from .exceptions import PageOutOfRangeError, PDFReadError
from .page import PDFPage


def _calculate_dpi_from_dimensions(width: int, default_dpi: int) -> int:
    """Calculate adjusted DPI based on page width.

    Reduces DPI for pages wider than 1000 points to prevent
    excessively large rendered images.

    Args:
        width: Page width in points
        default_dpi: Default DPI to use/adjust

    Returns:
        Adjusted DPI value
    """
    if width > 1000:
        return int(default_dpi / (width / 1000))
    return default_dpi


class PDFReader(PDFReaderBase):
    """
    A memory-efficient PDF reader that yields pages lazily.

    Uses FitzBackend by default for fast rendering, with automatic
    fallback to PdftoppmBackend if fitz fails to load the PDF.

    Example usage:
        # Memory-efficient iteration
        with PDFReader("large.pdf") as reader:
            for page in reader:
                image = page.to_image()  # Image loaded here
                process(image)
                # image goes out of scope, memory freed

        # Access specific pages
        with PDFReader("document.pdf") as reader:
            first_page = reader.get_page(0)
            image = first_page.to_image()

        # Iterate over a range of pages
        with PDFReader("document.pdf") as reader:
            for page in reader.iter_pages(start=5, end=10):
                image = page.to_image()
    """

    def __init__(
        self,
        path: str,
        dpi: int = 201,
        password: str = "",
        use_dynamic_dpi: bool = True,
    ):
        """
        Initialize the PDF reader.

        Args:
            path: Path to the PDF file
            dpi: Dots per inch for rendering (default: 201)
            password: Password for encrypted PDFs (optional)
            use_dynamic_dpi: Whether to dynamically adjust DPI based on page size
                            (default: True)
        """
        self._path = str(path)
        self._password = password
        self._closed = False

        # Create the backend with auto-fallback and calculate DPI
        # This opens the PDF only once instead of twice
        self._backend, self._dpi = self._create_backend(dpi, use_dynamic_dpi)

        logger.info(f"PDFReader initialized: path='{self._path}', pages={self.page_count}, dpi={self._dpi}")

    def _create_backend(self, default_dpi: int, use_dynamic_dpi: bool) -> Tuple[PDFBackend, int]:
        """
        Create a PDF backend with automatic fallback.

        Tries FitzBackend first (faster, pure Python), falls back to
        PdftoppmBackend if fitz fails. Also calculates DPI using the
        backend's page dimensions if use_dynamic_dpi is True.

        Args:
            default_dpi: Default DPI to use for rendering
            use_dynamic_dpi: Whether to adjust DPI based on page dimensions

        Returns:
            Tuple of (PDFBackend instance, calculated DPI)

        Raises:
            PDFReadError: If both backends fail
        """
        fitz_error: Optional[Exception] = None

        # Try FitzBackend first
        try:
            backend = FitzBackend(self._path, self._password)
            logger.info(f"Using FitzBackend for '{self._path}'")

            # Calculate DPI using backend's page dimensions (avoids reopening PDF)
            dpi = default_dpi
            if use_dynamic_dpi:
                try:
                    width, _ = backend.get_page_dimensions(0)
                    dpi = _calculate_dpi_from_dimensions(width, default_dpi)
                except Exception as e:
                    logger.warning(
                        f"Failed to calculate dynamic DPI for '{self._path}': {e}. Using default DPI: {default_dpi}"
                    )

            return backend, dpi
        except Exception as e:
            fitz_error = e
            logger.warning(f"FitzBackend failed for '{self._path}': {e}. Falling back to PdftoppmBackend.")

        # Fallback to PdftoppmBackend with default DPI
        # (can't calculate dynamic DPI since fitz failed)
        try:
            backend = PdftoppmBackend(self._path, default_dpi, self._password)
            logger.info(f"Using PdftoppmBackend (fallback) for '{self._path}'")
            return backend, default_dpi
        except Exception as e:
            logger.error(
                f"Both backends failed for '{self._path}'. FitzBackend error: {fitz_error}. PdftoppmBackend error: {e}."
            )
            raise PDFReadError(
                f"Failed to open PDF '{self._path}'. FitzBackend: {fitz_error}. PdftoppmBackend: {e}."
            ) from e

    @property
    def path(self) -> str:
        """Return the path to the PDF file."""
        return self._path

    @property
    def page_count(self) -> int:
        """Return the total number of pages."""
        return self._backend.page_count

    @property
    def dpi(self) -> int:
        """Return the DPI used for rendering."""
        return self._dpi

    def __len__(self) -> int:
        """Return the number of pages."""
        return self.page_count

    def __iter__(self) -> Iterator[PDFPage]:
        """
        Iterate over all pages in the PDF.

        Yields:
            PDFPage objects for each page
        """
        for i in range(self.page_count):
            yield PDFPage(self, i)

    def __getitem__(self, index: int) -> PDFPage:
        """
        Get a page by index.

        Args:
            index: Zero-based page index (supports negative indexing)

        Returns:
            PDFPage for the requested page

        Raises:
            PageOutOfRangeError: If the index is out of range
        """
        # Handle negative indexing
        if index < 0:
            index = self.page_count + index

        if index < 0 or index >= self.page_count:
            raise PageOutOfRangeError(index, self.page_count)

        return PDFPage(self, index)

    def get_page(self, index: int) -> PDFPage:
        """
        Get a page by index.

        Args:
            index: Zero-based page index

        Returns:
            PDFPage for the requested page

        Raises:
            PageOutOfRangeError: If the index is out of range
        """
        return self[index]

    def iter_pages(self, start: int = 0, end: Optional[int] = None) -> Iterator[PDFPage]:
        """
        Iterate over a range of pages.

        Args:
            start: Starting page index (inclusive, default: 0)
            end: Ending page index (exclusive, default: page_count)

        Yields:
            PDFPage objects for each page in the range
        """
        if end is None:
            end = self.page_count

        # Clamp values to valid range
        start = max(0, start)
        end = min(end, self.page_count)

        for i in range(start, end):
            yield PDFPage(self, i)

    def close(self) -> None:
        """Close the reader and release resources."""
        if not self._closed and self._backend:
            self._backend.close()
            self._closed = True
            logger.debug(f"PDFReader closed for '{self._path}'")

    def __enter__(self) -> "PDFReader":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and release resources."""
        self.close()

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()

    def __repr__(self) -> str:
        return f"PDFReader(path='{self._path}', pages={self.page_count}, dpi={self._dpi})"
