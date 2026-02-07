"""
Abstract base class for OCR implementations.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import pandas as pd

if TYPE_CHECKING:
    from ..pdf_reader import PDFPage, PDFReader


# Standard columns for OCR DataFrames
OCR_COLUMNS = [
    "Text",
    "x0",
    "y0",
    "x2",
    "y2",
    "block",
    "line",
    "space_type",
    "page",
    "confidence",
    "index_sort",
]


class OCRBase(ABC):
    """
    Abstract base class for OCR implementations.

    This class defines the interface that all OCR implementations must follow.
    Subclasses should implement process_page() to handle their specific OCR logic.

    Example:
        class MyOCR(OCRBase):
            def process_page(self, page):
                # Implementation here
                pass
    """

    @abstractmethod
    def process_page(self, page: "PDFPage", page_index: int = 0) -> pd.DataFrame:
        """
        Process a single page and extract OCR data.

        Args:
            page: PDFPage object containing the page to process
            page_index: Zero-based page index for the 'page' column

        Returns:
            pd.DataFrame with columns: Text, x0, y0, x2, y2, block, line,
                space_type, page, confidence, index_sort
        """
        pass

    def process_document(self, reader: "PDFReader") -> List[pd.DataFrame]:
        """
        Process all pages in a PDF document.

        Args:
            reader: PDFReader instance

        Returns:
            List of DataFrames, one per page
        """
        df_list = []
        for i, page in enumerate(reader):
            df = self.process_page(page, page_index=i)
            df_list.append(df)
        return df_list

    @staticmethod
    def create_empty_df() -> pd.DataFrame:
        """
        Create an empty DataFrame with standard OCR columns.

        Returns:
            Empty pd.DataFrame with OCR_COLUMNS
        """
        return pd.DataFrame(columns=OCR_COLUMNS)
