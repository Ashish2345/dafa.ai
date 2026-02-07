"""
parse_pdf - A memory-efficient PDF reader with lazy page loading.

This module provides an OOP-based PDF reader that yields images lazily
to prevent OOM issues when processing PDFs with many pages.

Example usage:
    from parse_pdf import PDFReader

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
"""

from .exceptions import EncryptedPDFError, PageOutOfRangeError, PDFReadError
from .page import PageScalar, PDFPage
from .reader import PDFReader

__all__ = [
    "PDFReader",
    "PDFPage",
    "PageScalar",
    "PDFReadError",
    "EncryptedPDFError",
    "PageOutOfRangeError",
]
