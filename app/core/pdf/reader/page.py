"""
PDFPage class for lazy image loading.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from pypdf import PdfReader

if TYPE_CHECKING:
    from .reader import PDFReader as PDFReaderClass


@dataclass(frozen=True)
class PageScalar:
    """Represents page dimensions and rotation."""

    width: int
    height: int
    angle: int  # rotation in degrees (0, 90, 180, 270)

    def to_dict(self) -> dict:
        """Return the page scalar as a dictionary."""
        return {"width": self.width, "height": self.height, "angle": self.angle}


@dataclass(frozen=True)
class Word:
    """Represents a word extracted from a PDF page with pixel coordinates."""

    x0: int
    y0: int
    x2: int
    y2: int
    text: str
    block: int
    line: int

    def to_dict(self) -> dict:
        """Return the word as a dictionary."""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x2": self.x2,
            "y2": self.y2,
            "text": self.text,
            "block": self.block,
            "line": self.line,
        }


@dataclass
class PageData:
    """Represents extracted page data including words, dimensions, and image."""

    words: List[Word]
    page_scalar: PageScalar
    image: np.ndarray


class PDFPage:
    """
    Represents a single page in a PDF document.

    This class provides lazy loading of page images - the image is only
    rendered when to_image() is called, preventing OOM issues with large PDFs.
    """

    def __init__(self, reader: "PDFReaderClass", index: int):
        """
        Initialize a PDFPage.

        Args:
            reader: The parent PDFReader instance
            index: Zero-based page index
        """
        self._reader = reader
        self._index = index
        self._orientation: Optional[int] = None
        self._data: Optional[PageData] = None

    @property
    def index(self) -> int:
        """Return the zero-based page index."""
        return self._index

    @property
    def page_number(self) -> int:
        """Return the one-based page number."""
        return self._index + 1

    @property
    def orientation(self) -> int:
        """
        Return the page orientation in degrees (0, 90, 180, or 270).

        Uses pypdf to extract the /Rotate metadata from the PDF.
        """
        if self._orientation is None:
            self._orientation = self._get_orientation()
        return self._orientation

    def _get_orientation(self) -> int:
        """
        Extract orientation from PDF metadata using pypdf.

        Returns:
            Orientation in degrees (0, 90, 180, or 270)
        """
        try:
            pdf = PdfReader(self._reader.path, strict=False)
            rotation = pdf.pages[self._index].get("/Rotate")
            return rotation if rotation in (0, 90, 180, 270) else 0
        except Exception:
            return 0

    @property
    def width(self) -> int:
        """
        Return the page width in points (1/72 inch).

        Note: This is the original PDF page width, not the rendered image width.
        """
        w, _ = self._reader._backend.get_page_dimensions(self._index)
        return w

    @property
    def height(self) -> int:
        """
        Return the page height in points (1/72 inch).

        Note: This is the original PDF page height, not the rendered image height.
        """
        _, h = self._reader._backend.get_page_dimensions(self._index)
        return h

    @property
    def page_scalar(self) -> PageScalar:
        """
        Return page dimensions and rotation as a single object.

        Returns:
            PageScalar with width, height, and angle (rotation in degrees)
        """
        return PageScalar(
            width=self.width,
            height=self.height,
            angle=self.orientation,
        )

    def to_image(self) -> np.ndarray:
        """
        Render the page to a numpy array.

        This is where the actual image rendering happens. The image is
        rendered on-demand and not cached, allowing memory to be freed
        when the returned array goes out of scope.

        Returns:
            numpy array in BGR format (OpenCV compatible)
        """
        return self._reader._backend.render_page(self._index, self._reader._dpi)

    @property
    def data(self) -> PageData:
        """
        Return page data containing words, dimensions, and rendered image.

        The words are Word objects with pixel coordinates scaled to the
        page dimensions. The image is rendered in BGR format (OpenCV compatible).
        The result is cached after first access.

        Returns:
            PageData with words (List[Word]), page_scalar, and image (np.ndarray)
        """
        if self._data is None:
            self._data = self._extract_data()
        return self._data

    def _extract_data(self) -> PageData:
        """Extract and process page data from the PDF using the backend."""
        # Render the page image FIRST to get pixel dimensions
        image = self.to_image()
        image_height, image_width = image.shape[:2]

        words: List[Word] = []

        # Get words from backend (uses already-open PDF)
        raw_words = self._reader._backend.get_page_words(self._index)

        if raw_words:
            # Get PDF page dimensions (in points) for scaling
            page_width, page_height = self._reader._backend.get_page_dimensions(self._index)

            for w in raw_words:
                x0, y0, x2, y2, text, block, line, _ = w

                # Scale from PDF points to image pixels
                words.append(
                    Word(
                        x0=int(round(x0 / page_width * image_width)),
                        y0=int(round(y0 / page_height * image_height)),
                        x2=int(round(x2 / page_width * image_width)),
                        y2=int(round(y2 / page_height * image_height)),
                        text=text,
                        block=block,
                        line=line,
                    )
                )

        return PageData(words=words, page_scalar=self.page_scalar, image=image)

    def __repr__(self) -> str:
        return f"PDFPage(index={self._index}, width={self.width}, height={self.height})"
