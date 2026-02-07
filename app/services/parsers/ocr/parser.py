"""
OCR Parser Component

Transforms raw OCR output into a clean, structured, and semantically consistent format
for downstream agents. This parser is responsible for:

1. Removing OCR noise and inconsistencies
2. Correctly grouping fields, tables, and multi-page content
3. Producing a deterministic, agent-friendly schema
"""

import re
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from docllm.parser.input_parser import get_input_parsing_method
from docllm.utils import combine, reset_lines


class OCRParser:
    """
    Parser component for cleaning and structuring OCR output.

    This parser transforms raw OCR DataFrames into a normalized format that:
    - Removes noise and inconsistencies
    - Groups related content (fields, tables, blocks)
    - Handles multi-page documents
    - Produces deterministic, agent-friendly output
    """

    def __init__(
        self,
        input_parsing_method: str = "layout_conserved_advance",
        min_confidence: float = 0.5,
        enable_table_detection: bool = True,
        enable_field_grouping: bool = True,
        combine_words: bool = True,
        gap_factor: float = 0.75,
    ):
        """
        Initialize the OCR parser.

        Args:
            input_parsing_method: Method for parsing OCR input (e.g., "layout_conserved_advance")
            min_confidence: Minimum confidence threshold for OCR words (0.0-1.0)
            enable_table_detection: Whether to detect and extract tables
            enable_field_grouping: Whether to group fields and key-value pairs
            combine_words: Whether to combine close words into phrases
            gap_factor: Factor for determining word proximity when combining (0.0-1.0)
        """
        self.input_parsing_method = input_parsing_method
        self.min_confidence = min_confidence
        self.enable_table_detection = enable_table_detection
        self.enable_field_grouping = enable_field_grouping
        self.combine_words = combine_words
        self.gap_factor = gap_factor

    def parse(
        self,
        raw_ocr: List[pd.DataFrame],
        page_scalars: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Parse raw OCR output into structured format.

        Args:
            raw_ocr: List of DataFrames, one per page, with OCR results
            page_scalars: Optional list of page metadata (width, height, angle)

        Returns:
            Dictionary with structured parsed data:
            {
                "cleaned_ocr": List[pd.DataFrame],  # Cleaned OCR data per page
                "pages": List[Dict],  # Structured page data
                "tables": List[Dict],  # Detected tables
                "fields": Dict,  # Grouped fields/key-value pairs
                "content": str,  # Full document text
                "metadata": Dict,  # Parsing metadata
            }
        """
        if not raw_ocr:
            logger.warning("Empty OCR input provided")
            return self._empty_result()

        logger.info(f"Parsing {len(raw_ocr)} pages of OCR data")

        # Step 1: Clean and normalize OCR data
        cleaned_ocr = self._clean_ocr_data(raw_ocr)

        # Step 2: Combine words if enabled
        if self.combine_words:
            cleaned_ocr = self._combine_words(cleaned_ocr)

        # Step 3: Extract structured content per page
        pages_data = []
        all_tables = []
        all_fields = {}
        all_content_parts = []

        for page_idx, ocr_df in enumerate(cleaned_ocr):
            page_number = page_idx + 1
            page_scalar = page_scalars[page_idx] if page_scalars and page_idx < len(page_scalars) else {}

            # Extract page content
            page_content = self._extract_page_content(ocr_df)
            all_content_parts.append(page_content)

            # Detect tables
            tables = []
            if self.enable_table_detection:
                tables = self._detect_tables(ocr_df, page_number, page_scalar)
                all_tables.extend(tables)

            # Group fields
            fields = {}
            if self.enable_field_grouping:
                fields = self._group_fields(ocr_df, page_number)

            # Create page data structure
            page_data = {
                "page_number": page_number,
                "content": page_content,
                "blocks": self._extract_blocks(ocr_df, page_number, page_scalar),
                "tables": tables,
                "fields": fields,
                "word_count": len(ocr_df) if not ocr_df.empty else 0,
                "bbox": self._calculate_page_bbox(ocr_df, page_scalar),
            }
            pages_data.append(page_data)

            # Merge fields across pages
            for section, section_fields in fields.items():
                if section not in all_fields:
                    all_fields[section] = {}
                all_fields[section].update(section_fields)

        # Step 4: Generate full document content
        full_content = "\n\n".join(all_content_parts)

        # Step 5: Build metadata
        metadata = {
            "total_pages": len(cleaned_ocr),
            "total_words": sum(len(df) for df in cleaned_ocr if not df.empty),
            "total_tables": len(all_tables),
            "total_fields": sum(len(section) for section in all_fields.values()),
            "parsing_method": self.input_parsing_method,
        }

        result = {
            "cleaned_ocr": cleaned_ocr,
            "pages": pages_data,
            "tables": all_tables,
            "fields": all_fields,
            "content": full_content,
            "metadata": metadata,
        }

        logger.info(
            f"Parsing complete: {metadata['total_pages']} pages, "
            f"{metadata['total_words']} words, {metadata['total_tables']} tables, "
            f"{metadata['total_fields']} fields"
        )

        return result

    def _clean_ocr_data(self, raw_ocr: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Clean and normalize OCR data by removing noise and inconsistencies.

        Args:
            raw_ocr: List of raw OCR DataFrames

        Returns:
            List of cleaned OCR DataFrames
        """
        cleaned = []

        for page_idx, df in enumerate(raw_ocr):
            if df is None or df.empty:
                logger.debug(f"Page {page_idx + 1}: Empty OCR data")
                cleaned.append(pd.DataFrame())
                continue

            # Make a copy to avoid modifying original
            df_clean = df.copy()

            # Ensure required columns exist
            required_cols = ["Text", "x0", "y0", "x2", "y2"]
            missing_cols = [col for col in required_cols if col not in df_clean.columns]
            if missing_cols:
                logger.warning(f"Page {page_idx + 1}: Missing columns {missing_cols}, skipping")
                cleaned.append(pd.DataFrame())
                continue

            # Remove rows with empty or NaN text
            df_clean = df_clean.dropna(subset=["Text"])
            df_clean = df_clean[df_clean["Text"].astype(str).str.strip() != ""]

            # Filter by confidence if available
            if "confidence" in df_clean.columns:
                df_clean = df_clean[df_clean["confidence"] >= self.min_confidence]

            # Clean text: remove common OCR artifacts
            df_clean["Text"] = df_clean["Text"].astype(str).apply(self._clean_text)

            # Ensure numeric columns are numeric
            for col in ["x0", "y0", "x2", "y2", "block", "line", "page", "confidence"]:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

            # Remove invalid bounding boxes
            df_clean = df_clean[
                (df_clean["x2"] > df_clean["x0"]) & (df_clean["y2"] > df_clean["y0"])
            ]

            # Ensure page numbers are set
            if "page" not in df_clean.columns or df_clean["page"].isna().all():
                df_clean["page"] = page_idx

            # Reset lines for better grouping
            if "line" not in df_clean.columns or df_clean["line"].isna().all():
                df_clean = pd.DataFrame(reset_lines(df_clean))

            cleaned.append(df_clean)

        return cleaned

    def _clean_text(self, text: str) -> str:
        """
        Clean individual text strings to remove OCR noise.

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove common OCR artifacts
        text = re.sub(r"[^\S\n]+", " ", text)  # Normalize whitespace
        text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
        text = text.strip()

        # Remove common OCR errors (can be extended)
        # Remove isolated punctuation that's likely noise
        text = re.sub(r"^\W+$", "", text)

        return text

    def _combine_words(self, ocr_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Combine close words into phrases.

        Args:
            ocr_list: List of OCR DataFrames

        Returns:
            List of DataFrames with combined words
        """
        combined = []

        for df in ocr_list:
            if df is None or df.empty:
                combined.append(df)
                continue

            try:
                df_combined = combine(df, gap_factor=self.gap_factor)
                combined.append(df_combined)
            except (ValueError, AttributeError, KeyError) as e:
                logger.warning(f"Error combining words: {e}, using original data")
                combined.append(df)

        return combined

    def _extract_page_content(self, ocr_df: pd.DataFrame) -> str:
        """
        Extract text content from OCR DataFrame using configured parsing method.

        Args:
            ocr_df: OCR DataFrame for a single page

        Returns:
            Extracted text content
        """
        if ocr_df is None or ocr_df.empty:
            return ""

        try:
            parsing_func = get_input_parsing_method(self.input_parsing_method)
            text, _ = parsing_func(ocr_df)
            return text
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Error extracting content with method {self.input_parsing_method}: {e}")
            # Fallback to simple line-based extraction
            return self._extract_simple_text(ocr_df)

    def _extract_simple_text(self, ocr_df: pd.DataFrame) -> str:
        """
        Simple fallback text extraction method.

        Args:
            ocr_df: OCR DataFrame

        Returns:
            Extracted text
        """
        if ocr_df.empty:
            return ""

        # Group by line and sort
        lines = {}
        for _, row in ocr_df.iterrows():
            line_num = int(row.get("line", 0))
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append((row.get("x0", 0), row.get("Text", "")))

        # Sort words within each line by x0
        text_parts = []
        for line_num in sorted(lines.keys()):
            line_words = sorted(lines[line_num], key=lambda x: x[0])
            line_text = " ".join(word for _, word in line_words if word.strip())
            if line_text:
                text_parts.append(line_text)

        return "\n".join(text_parts)

    def _detect_tables(
        self, ocr_df: pd.DataFrame, page_number: int, page_scalar: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect tables in OCR data.

        Args:
            ocr_df: OCR DataFrame for a page
            page_number: Page number
            page_scalar: Page dimensions

        Returns:
            List of detected tables with structure information
        """
        if ocr_df.empty:
            return []

        tables = []

        try:
            # Simple table detection: look for aligned columns
            # Group by block and analyze structure
            blocks = {}
            for _, row in ocr_df.iterrows():
                block_id = int(row.get("block", 0))
                if block_id not in blocks:
                    blocks[block_id] = []
                blocks[block_id].append(row)

            # Analyze each block for table-like structure
            for block_id, block_rows in blocks.items():
                if len(block_rows) < 3:  # Need at least a few rows for a table
                    continue

                # Check for column alignment (similar x0 values)
                x0_values = [row.get("x0", 0) for row in block_rows]
                x0_std = pd.Series(x0_values).std()

                # If x0 values are relatively consistent, might be a table
                if x0_std < 50:  # Threshold for column alignment
                    # Group by line to get rows
                    rows_by_line = {}
                    for row in block_rows:
                        line_num = int(row.get("line", 0))
                        if line_num not in rows_by_line:
                            rows_by_line[line_num] = []
                        rows_by_line[line_num].append(row)

                    if len(rows_by_line) >= 2:  # At least 2 rows
                        # Calculate table bbox
                        all_x0 = [r.get("x0", 0) for r in block_rows]
                        all_y0 = [r.get("y0", 0) for r in block_rows]
                        all_x2 = [r.get("x2", 0) for r in block_rows]
                        all_y2 = [r.get("y2", 0) for r in block_rows]

                        page_width = page_scalar.get("width", 1)
                        page_height = page_scalar.get("height", 1)

                        table_bbox = [
                            min(all_x0) * page_width,
                            min(all_y0) * page_height,
                            max(all_x2) * page_width,
                            max(all_y2) * page_height,
                        ]

                        table = {
                            "table_id": f"table_{page_number}_{block_id}",
                            "page": page_number,
                            "block": block_id,
                            "bbox": table_bbox,
                            "row_count": len(rows_by_line),
                            "column_count": self._estimate_column_count(block_rows),
                        }
                        tables.append(table)

        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Error detecting tables on page {page_number}: {e}")

        return tables

    def _estimate_column_count(self, block_rows: List[pd.Series]) -> int:
        """
        Estimate number of columns in a table block.

        Args:
            block_rows: Rows in the block

        Returns:
            Estimated column count
        """
        if not block_rows:
            return 0

        # Group x0 positions into columns
        x0_positions = sorted(set(row.get("x0", 0) for row in block_rows))
        if not x0_positions:
            return 0

        # Cluster nearby x0 positions
        columns = []
        threshold = 30  # Pixels

        for x0 in x0_positions:
            if not columns:
                columns.append([x0])
            else:
                # Check if x0 is close to any existing column
                added = False
                for col in columns:
                    if abs(x0 - col[0]) < threshold:
                        col.append(x0)
                        added = True
                        break
                if not added:
                    columns.append([x0])

        return len(columns)

    def _group_fields(
        self, ocr_df: pd.DataFrame, page_number: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group fields and key-value pairs from OCR data.

        Args:
            ocr_df: OCR DataFrame
            page_number: Page number

        Returns:
            Dictionary of grouped fields by section
        """
        if ocr_df.empty:
            return {}

        fields = {"general": {}}

        try:
            # Look for key-value patterns (e.g., "Label: Value")
            kv_pattern = re.compile(r"^(.+?):\s*(.+)$")

            for _, row in ocr_df.iterrows():
                text = str(row.get("Text", "")).strip()
                match = kv_pattern.match(text)

                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()

                    if key and value:
                        # Calculate bbox
                        page_width = 1  # Will be normalized later
                        page_height = 1

                        fields["general"][key.lower()] = {
                            "value": value,
                            "key": key,
                            "bbox": [
                                row.get("x0", 0) * page_width,
                                row.get("y0", 0) * page_height,
                                row.get("x2", 0) * page_width,
                                row.get("y2", 0) * page_height,
                            ],
                            "page": page_number,
                            "confidence": row.get("confidence", 1.0),
                        }

        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Error grouping fields on page {page_number}: {e}")

        return fields

    def _extract_blocks(
        self, ocr_df: pd.DataFrame, page_number: int, page_scalar: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract content blocks from OCR data.

        Args:
            ocr_df: OCR DataFrame
            page_number: Page number
            page_scalar: Page dimensions

        Returns:
            List of block dictionaries
        """
        if ocr_df.empty:
            return []

        blocks = []

        try:
            # Group by block
            block_rows = {}
            for _, row in ocr_df.iterrows():
                block_id = int(row.get("block", 0))
                if block_id not in block_rows:
                    block_rows[block_id] = []
                block_rows[block_id].append(row)

            page_width = page_scalar.get("width", 1)
            page_height = page_scalar.get("height", 1)

            for block_id, rows in sorted(block_rows.items()):
                # Calculate block bbox
                x0 = min(r.get("x0", 0) for r in rows) * page_width
                y0 = min(r.get("y0", 0) for r in rows) * page_height
                x2 = max(r.get("x2", 0) for r in rows) * page_width
                y2 = max(r.get("y2", 0) for r in rows) * page_height

                # Extract block text
                block_text = self._extract_block_text(rows)

                block = {
                    "block_id": f"block_{page_number}_{block_id}",
                    "content": block_text,
                    "bbox": [x0, y0, x2, y2],
                    "word_count": len(rows),
                }
                blocks.append(block)

        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Error extracting blocks on page {page_number}: {e}")

        return blocks

    def _extract_block_text(self, rows: List[pd.Series]) -> str:
        """
        Extract text from a list of rows.

        Args:
            rows: List of row Series

        Returns:
            Extracted text
        """
        # Group by line
        lines = {}
        for row in rows:
            line_num = int(row.get("line", 0))
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append((row.get("x0", 0), row.get("Text", "")))

        # Sort and join
        text_parts = []
        for line_num in sorted(lines.keys()):
            line_words = sorted(lines[line_num], key=lambda x: x[0])
            line_text = " ".join(word for _, word in line_words if word.strip())
            if line_text:
                text_parts.append(line_text)

        return "\n".join(text_parts)

    def _calculate_page_bbox(
        self, ocr_df: pd.DataFrame, page_scalar: Dict[str, Any]
    ) -> List[float]:
        """
        Calculate bounding box for entire page content.

        Args:
            ocr_df: OCR DataFrame
            page_scalar: Page dimensions

        Returns:
            Bounding box [x0, y0, x2, y2]
        """
        if ocr_df.empty:
            page_width = page_scalar.get("width", 0)
            page_height = page_scalar.get("height", 0)
            return [0.0, 0.0, float(page_width), float(page_height)]

        page_width = page_scalar.get("width", 1)
        page_height = page_scalar.get("height", 1)

        x0 = float(ocr_df["x0"].min() * page_width)
        y0 = float(ocr_df["y0"].min() * page_height)
        x2 = float(ocr_df["x2"].max() * page_width)
        y2 = float(ocr_df["y2"].max() * page_height)

        return [x0, y0, x2, y2]

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "cleaned_ocr": [],
            "pages": [],
            "tables": [],
            "fields": {},
            "content": "",
            "metadata": {
                "total_pages": 0,
                "total_words": 0,
                "total_tables": 0,
                "total_fields": 0,
                "parsing_method": self.input_parsing_method,
            },
        }
