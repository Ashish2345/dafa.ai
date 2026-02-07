"""
Digital OCR implementation for extracting text from digitally-created PDFs.
"""

from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from ...reader import PDFPage, PDFReader
from ..base import OCR_COLUMNS, OCRBase
from ..utils import break_colons, remove_no_break_space, sort_df


class DigitalOCR(OCRBase):
    """
    OCR implementation for digitally-created PDFs.

    Digital PDFs contain embedded text that can be extracted directly from
    the PDF structure. This class converts the extracted words into a
    standardized DataFrame format.

    Args:
        reset_lines_and_sort: bool
            If True, recalculates line assignments and sorts the DataFrame.
            Recommended for digital documents. Default: True
        gap_factor: float
            Factor used for word combination (reserved for future use).
            Default: 0.75

    Example:
        from diu_new.pdf_reader import PDFReader
        from diu_new.ocr import DigitalOCR

        ocr = DigitalOCR(reset_lines_and_sort=True)

        with PDFReader("document.pdf") as reader:
            # Process all pages
            df_list = ocr.process_document(reader)

            # Or process a single page
            df = ocr.process_page(reader[0])
    """

    def __init__(self, reset_lines_and_sort: bool = True, gap_factor: float = 0.75):
        self.reset_lines_and_sort = reset_lines_and_sort
        self.gap_factor = gap_factor

    def process_page(self, page: PDFPage, page_index: int = 0) -> pd.DataFrame:
        """
        Process a single PDF page and extract text as a DataFrame.

        Args:
            page: PDFPage object to process
            page_index: Zero-based page index for the 'page' column

        Returns:
            pd.DataFrame with standardized OCR columns
        """
        # Get page data (words, dimensions, image)
        page_data = page.data
        words = page_data.words
        page_scalar = page_data.page_scalar
        image = page_data.image

        if not words:
            logger.debug(f"No words found on page {page_index}")
            return self.create_empty_df()

        # Get image dimensions for coordinate normalization
        image_height, image_width = image.shape[:2]

        # Convert words to DataFrame
        df = self._words_to_df(words, page_index, image_width, image_height)

        # Handle page orientation
        df = self._orient_df(df, page_scalar.angle)

        # Optionally reset lines and sort
        if self.reset_lines_and_sort:
            df = sort_df(df)

        return df

    def _words_to_df(
        self,
        words: List,
        page_index: int,
        image_width: int,
        image_height: int,
    ) -> pd.DataFrame:
        """
        Convert Word objects to a DataFrame.

        Args:
            words: List of Word objects from PDFPage.data
            page_index: Page number for the 'page' column
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            pd.DataFrame with OCR data
        """
        data = []

        for i, word in enumerate(words):
            text = remove_no_break_space(word.text)
            if not text:
                continue

            # Get normalized coordinates (0-1 range)
            x0 = word.x0 / image_width if image_width > 0 else 0
            y0 = word.y0 / image_height if image_height > 0 else 0
            x2 = word.x2 / image_width if image_width > 0 else 0
            y2 = word.y2 / image_height if image_height > 0 else 0

            # Determine space_type using break_colons logic
            # Check if this is the last word or if next word is on different line/block
            try:
                next_word = words[i + 1]
                if next_word.block != word.block or next_word.line != word.line:
                    space_type = 2  # Line break
                elif next_word.text.startswith(":"):
                    space_type = 2
                else:
                    space_type = 1  # Same line
            except IndexError:
                space_type = 2  # End of page

            # Handle colons in text
            for txt, box, brk in break_colons(text, (x0, y0, x2, y2), space_type):
                _x0, _y0, _x2, _y2 = box
                data.append(
                    {
                        "Text": txt,
                        "x0": _x0,
                        "y0": _y0,
                        "x2": _x2,
                        "y2": _y2,
                        "block": word.block,
                        "line": word.line,
                        "space_type": brk,
                        "page": page_index,
                        "confidence": 0.99,  # High confidence for digital text
                        "index_sort": len(data),
                    }
                )

        df = pd.DataFrame(data, columns=OCR_COLUMNS)

        # Clean up text
        df["Text"] = df["Text"].str.replace("\n", "")
        df = df[df["Text"] != ""]

        return df

    def _orient_df(self, df: pd.DataFrame, orientation: int) -> pd.DataFrame:
        """
        Adjust DataFrame coordinates based on page orientation.

        Transforms coordinates from the given orientation to standard
        (0 degrees) orientation, assuming coordinates are normalized (0-1).

        Args:
            df: DataFrame with normalized coordinates
            orientation: Page rotation in degrees (0, 90, 180, 270)

        Returns:
            DataFrame with adjusted coordinates
        """
        if df.empty or orientation == 0:
            return df

        df = df.copy()

        try:
            logger.debug(f"Adjusting for orientation: {orientation}")

            if orientation in (270, 180):
                # Reflect along x and y axis
                df[["x0", "x2"]] = 1 - df[["x2", "x0"]].values
                df[["y0", "y2"]] = 1 - df[["y2", "y0"]].values

            if orientation in (90, 270):
                # Rotate from 90 to 0: swap x and y, adjust for rotation
                old_x0 = df["x0"].copy()
                old_x2 = df["x2"].copy()
                old_y0 = df["y0"].copy()
                old_y2 = df["y2"].copy()

                df["y0"] = old_x0
                df["y2"] = old_x2
                df["x0"] = 1 - old_y2
                df["x2"] = 1 - old_y0

        except Exception as e:
            logger.error(f"Error in orientation adjustment: {e}")

        return df

    def process_document(self, reader: PDFReader) -> List[pd.DataFrame]:
        """
        Process all pages in a PDF document.

        Args:
            reader: PDFReader instance

        Returns:
            List of DataFrames, one per page
        """
        logger.info(f"Processing digital PDF: {reader.path} ({reader.page_count} pages)")
        return super().process_document(reader)
