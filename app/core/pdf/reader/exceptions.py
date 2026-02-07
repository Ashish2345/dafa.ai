"""
Custom exceptions for the PDF reader module.
"""


class PDFReadError(Exception):
    """Base exception for PDF reading errors."""

    pass


class EncryptedPDFError(PDFReadError):
    """Raised when a PDF is encrypted and the password is invalid or missing."""

    pass


class PageOutOfRangeError(PDFReadError):
    """Raised when trying to access a page index that doesn't exist."""

    def __init__(self, index: int, page_count: int):
        self.index = index
        self.page_count = page_count
        super().__init__(
            f"Page index {index} is out of range. PDF has {page_count} pages (0-{page_count - 1})."
        )
