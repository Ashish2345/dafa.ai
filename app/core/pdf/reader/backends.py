"""
PDF rendering backend implementations.

Provides FitzBackend (primary) and PdftoppmBackend (fallback) for converting
PDF pages to images.
"""

import subprocess
import tempfile
from glob import glob
from pathlib import Path

import cv2
import fitz
import numpy as np
from loguru import logger

from .base import PDFBackend
from .exceptions import EncryptedPDFError, PageOutOfRangeError


class FitzBackend(PDFBackend):
    """
    PDF rendering backend using fitz (PyMuPDF).

    This is the primary backend - it's fast and doesn't require subprocess calls.
    May fail on some malformed PDFs, in which case PdftoppmBackend should be used.
    """

    def __init__(self, path: str, password: str = ""):
        """
        Initialize the fitz backend.

        Args:
            path: Path to the PDF file
            password: Password for encrypted PDFs (optional)

        Raises:
            EncryptedPDFError: If the PDF is encrypted and password is invalid
            PDFReadError: If the PDF cannot be opened
        """
        self._path = path
        self._doc = fitz.open(path)

        if self._doc.is_encrypted:
            if not self._doc.authenticate(password):
                self._doc.close()
                raise EncryptedPDFError(f"PDF '{path}' is encrypted and the provided password is invalid.")

        logger.debug(f"FitzBackend initialized for '{path}' with {len(self._doc)} pages")

    def render_page(self, index: int, dpi: int) -> np.ndarray:
        """
        Render a page to a numpy array using fitz.

        Args:
            index: Zero-based page index
            dpi: Dots per inch for rendering

        Returns:
            numpy array in BGR format (OpenCV compatible)

        Raises:
            PageOutOfRangeError: If the page index is invalid
        """
        if index < 0 or index >= len(self._doc):
            raise PageOutOfRangeError(index, len(self._doc))

        page = self._doc.load_page(index)

        # Create transformation matrix for the desired DPI
        # PDF default is 72 DPI, so we scale accordingly
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)

        # Render to pixmap
        pix = page.get_pixmap(matrix=matrix)

        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Convert RGB to BGR for OpenCV compatibility
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

        return img

    @property
    def page_count(self) -> int:
        """Return the total number of pages."""
        return len(self._doc)

    def get_page_dimensions(self, index: int) -> tuple[int, int]:
        """
        Get the dimensions of a page in points.

        Args:
            index: Zero-based page index

        Returns:
            Tuple of (width, height) in points
        """
        if index < 0 or index >= len(self._doc):
            raise PageOutOfRangeError(index, len(self._doc))

        page = self._doc.load_page(index)
        rect = page.rect
        return int(rect.width), int(rect.height)

    def close(self) -> None:
        """Close the PDF document and release resources."""
        if self._doc:
            self._doc.close()
            self._doc = None
            logger.debug(f"FitzBackend closed for '{self._path}'")

    def get_page_words(self, index: int) -> list:
        """Get words from a page using the already-open fitz document."""
        if index < 0 or index >= len(self._doc):
            raise PageOutOfRangeError(index, len(self._doc))

        page = self._doc.load_page(index)
        return page.get_text("words")


class PdftoppmBackend(PDFBackend):
    """
    PDF rendering backend using pdftoppm subprocess.

    This is the fallback backend - more robust for malformed PDFs but slower
    due to subprocess overhead and temp file I/O.
    """

    def __init__(self, path: str, dpi: int, password: str = ""):
        """
        Initialize the pdftoppm backend.

        Converts all pages to temporary JPEG files upfront.

        Args:
            path: Path to the PDF file
            dpi: Dots per inch for rendering
            password: Password for encrypted PDFs (optional)

        Raises:
            PDFReadError: If pdftoppm fails to convert the PDF
        """
        self._path = path
        self._dpi = dpi
        self._tempdir = tempfile.TemporaryDirectory()
        self._image_paths: list[str] = []

        # Build pdftoppm command
        command = [
            "pdftoppm",
            str(path),
            f"{self._tempdir.name}/page",
            "-jpeg",
            "-r",
            str(dpi),
        ]

        # Add password arguments if provided
        if password:
            command.extend(["-upw", str(password), "-opw", str(password)])

        logger.debug(f"Running pdftoppm command: {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            self._tempdir.cleanup()
            # Check if it's a password error
            if "Incorrect password" in e.stderr or "password" in e.stderr.lower():
                raise EncryptedPDFError(f"PDF '{path}' is encrypted and the provided password is invalid.") from e
            raise

        # Get sorted list of generated image files
        self._image_paths = sorted(glob(f"{self._tempdir.name}/*.jpg"))

        if not self._image_paths:
            self._tempdir.cleanup()
            raise RuntimeError(f"pdftoppm produced no output for '{path}'")

        logger.debug(f"PdftoppmBackend initialized for '{path}' with {len(self._image_paths)} pages")

    def render_page(self, index: int, dpi: int) -> np.ndarray:
        """
        Load a pre-rendered page from the temp directory.

        Note: The dpi parameter is ignored since pages were rendered at
        initialization time with a fixed DPI.

        Args:
            index: Zero-based page index
            dpi: Ignored (pages already rendered at init DPI)

        Returns:
            numpy array in BGR format (OpenCV compatible)

        Raises:
            PageOutOfRangeError: If the page index is invalid
        """
        if index < 0 or index >= len(self._image_paths):
            raise PageOutOfRangeError(index, len(self._image_paths))

        img = cv2.imread(self._image_paths[index])
        if img is None:
            raise RuntimeError(f"Failed to read image: {self._image_paths[index]}")
        return img

    @property
    def page_count(self) -> int:
        """Return the total number of pages."""
        return len(self._image_paths)

    def get_page_dimensions(self, index: int) -> tuple[int, int]:
        """
        Get the dimensions of a rendered page image.

        Args:
            index: Zero-based page index

        Returns:
            Tuple of (width, height) in pixels at the rendered DPI
        """
        if index < 0 or index >= len(self._image_paths):
            raise PageOutOfRangeError(index, len(self._image_paths))

        # Read just the image header to get dimensions without loading full image
        img = cv2.imread(self._image_paths[index])
        if img is None:
            raise RuntimeError(f"Failed to read image: {self._image_paths[index]}")
        return img.shape[1], img.shape[0]  # width, height

    def close(self) -> None:
        """Clean up temporary directory."""
        if self._tempdir:
            self._tempdir.cleanup()
            self._tempdir = None
            logger.debug(f"PdftoppmBackend closed for '{self._path}'")

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.close()

    def get_page_words(self, index: int) -> list:
        """PdftoppmBackend does not support word extraction."""
        return []
