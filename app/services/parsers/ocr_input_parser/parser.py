"""
OCR Input Parser Component

Transforms raw OCR output into a clean, structured, and semantically consistent format
for downstream agents. This parser is responsible for:

1. Removing OCR noise and inconsistencies
2. Correctly grouping fields, tables, and multi-page content
3. Producing a deterministic, agent-friendly schema

For regular content: Uses layout_conserved_with_lineno
For tables: Uses AWS Textract table extraction and outputs HTML
"""

import os
import re
from collections import defaultdict, namedtuple
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from loguru import logger

# ============================================================================
# Copied utility functions from docllm (standalone implementation)
# ============================================================================

# OCR Row structure
OCR_FIELDS = [
    "index_sort",
    "page",
    "block",
    "line",
    "x0",
    "y0",
    "x2",
    "y2",
    "Text",
    "space_type",
    "confidence",
]

OCRRow = namedtuple("OCRRow", OCR_FIELDS)


def _combine_list(dflist: list, gap_factor: float = 0.75) -> list:
    """
    Groups close words in a line.
    Copied from docllm.utils.utils
    """
    dfdata = []
    row, weights = defaultdict(list), []
    for i, w in enumerate(dflist):
        row["index_sort"] = w.Index
        row["page"] = w.page
        row["block"] = w.block
        row["line"] = w.line
        row["Text"].append(w.Text)
        row["x0"].append(w.x0)
        row["y0"].append(w.y0)
        row["x2"].append(w.x2)
        row["y2"].append(w.y2)
        row["space_type"] = w.space_type
        try:
            row["confidence"].append(w.confidence)
        except AttributeError:
            row["confidence"].append(1.0)
        weights.append(len(w.Text))

        if w.space_type > 1:
            row["Text"] = " ".join(row["Text"])
            row["x0"] = min(row["x0"])
            row["y0"] = min(row["y0"])
            row["x2"] = max(row["x2"])
            row["y2"] = max(row["y2"])
            try:
                row["confidence"] = np.average(row["confidence"], weights=weights)
            except Exception:
                row["confidence"] = 0.99
            dfdata.append(row)
            row, weights = defaultdict(list), []
            continue

        if w.Text.strip().endswith(":"):
            row["Text"] = " ".join(row["Text"])
            row["x0"] = min(row["x0"])
            row["y0"] = min(row["y0"])
            row["x2"] = max(row["x2"])
            row["y2"] = max(row["y2"])
            try:
                row["confidence"] = np.average(row["confidence"], weights=weights)
            except Exception:
                row["confidence"] = 0.99
            dfdata.append(row)
            row, weights = defaultdict(list), []
            continue

        try:
            nw = dflist[i + 1]
            if (nw.page != w.page) or (nw.block != w.block) or (nw.line != w.line):
                row["Text"] = " ".join(row["Text"])
                row["x0"] = min(row["x0"])
                row["y0"] = min(row["y0"])
                row["x2"] = max(row["x2"])
                row["y2"] = max(row["y2"])
                try:
                    row["confidence"] = np.average(row["confidence"], weights=weights)
                except Exception:
                    row["confidence"] = 0.99
                dfdata.append(row)
                row, weights = defaultdict(list), []
                continue

            h = w.y2 - w.y0
            nh = nw.y2 - nw.y0
            gap = nw.x0 - w.x2
            if gap > gap_factor * max(h, nh):
                row["Text"] = " ".join(row["Text"])
                row["x0"] = min(row["x0"])
                row["y0"] = min(row["y0"])
                row["x2"] = max(row["x2"])
                row["y2"] = max(row["y2"])
                try:
                    row["confidence"] = np.average(row["confidence"], weights=weights)
                except Exception:
                    row["confidence"] = 0.99
                dfdata.append(row)
                row, weights = defaultdict(list), []
                continue
        except IndexError:
            pass

    if row:
        row["Text"] = " ".join(row["Text"])
        row["x0"] = min(row["x0"])
        row["y0"] = min(row["y0"])
        row["x2"] = max(row["x2"])
        row["y2"] = max(row["y2"])
        try:
            row["confidence"] = np.average(row["confidence"], weights=weights)
        except Exception:
            row["confidence"] = 0.99
        dfdata.append(row)

    return [OCRRow(**w) for w in dfdata]


def combine(df: pd.DataFrame, gap_factor: float = 0.75) -> pd.DataFrame:
    """
    Clubs words belonging to same sentence.
    Copied from docllm.utils.utils
    """
    if df is None or df.empty:
        return df

    dflist = list(df.sort_index().itertuples())
    dflist = _combine_list(dflist, gap_factor)
    return pd.DataFrame(dflist)


def reset_lines(df_word_level: pd.DataFrame, factor: int = 1) -> List[Dict]:
    """
    Assigns the text lying in similar y-range to a same line.
    Copied from docllm.utils.utils
    """
    if df_word_level is None or df_word_level.empty:
        return []

    df_word_level_lst = list(df_word_level.itertuples())
    th = median([w.y2 - w.y0 for w in df_word_level_lst]) * factor
    rows_dict = defaultdict(list)
    row_key = 0
    for word in sorted(df_word_level_lst, key=lambda w: w.y0):
        if word.y0 - row_key >= 0.3 * th:
            row_key = word.y0
        rows_dict[row_key].append(word)
    rows_list = [sorted(r, key=lambda x: x.x0) for r in rows_dict.values()]

    items_list = []
    for line, (sublist, r) in enumerate(zip(rows_list, rows_dict)):
        for item in sublist:
            item_dict = item._asdict()
            item_dict["line"] = line
            item_dict["y0"] = r
            items_list.append(item_dict)

    return items_list


def _get_layout_conserved_text_with_lineno(df: pd.DataFrame, **kwargs) -> Tuple[str, pd.DataFrame]:
    """
    Converting pdf to text using layout conservation with line numbers.
    Simplified version based on docllm.parser.table_parser.get_layout_conserved_text_with_lineno
    """
    reset_lines_flag = kwargs.get("reset_lines", True)
    pixel_to_char = kwargs.get("pixel_to_char", 0.15)
    max_spaces = kwargs.get("size", 1000)
    line_num_at_start = kwargs.get("line_num_at_start", True)
    line_num_start_delim = kwargs.get("line_num_start_delim", "->")
    line_num_end_delim = kwargs.get("line_num_end_delim", "<-")
    texts: List[str] = []

    if df.empty:
        logger.warning("[WARNING] Empty DataFrame provided for text parsing.")
        return ("", df)

    # Group words by page and line
    all_lines = []
    for pg_no in sorted(df["page"].unique()):
        filtered_df = df[df["page"] == pg_no]

        if reset_lines_flag:
            filtered_df = pd.DataFrame(reset_lines(filtered_df))

        # Group by line
        for line_num in sorted(filtered_df["line"].unique()):
            line_df = filtered_df[filtered_df["line"] == line_num].sort_values("x0")
            if not line_df.empty:
                all_lines.append({
                    "words": line_df,
                    "x0": line_df["x0"].min(),
                    "x2": line_df["x2"].max(),
                    "y0": line_df["y0"].min(),
                    "y2": line_df["y2"].max(),
                })

    # Calculate max spaces needed
    for line in all_lines:
        line_x2 = line["x2"]
        max_spaces = max(max_spaces, round(line_x2 * pixel_to_char))

    line_no_length = len(str(len(all_lines)))

    # Build text with line numbers
    for line_no, line in enumerate(all_lines):
        final_string = [" "] * max_spaces
        block_wise_indices = defaultdict(lambda: "")

        for _, word_row in line["words"].iterrows():
            incoming_line = int(word_row.get("line", 0))
            incoming_block = int(word_row.get("block", 0))
            start_index = round((word_row["x0"] - 0) * pixel_to_char)
            word_text = str(word_row["Text"]).rstrip().lstrip()
            len_word = len(word_text)

            line_block_idx = f"line_{incoming_line}_block_{incoming_block}"

            if block_wise_indices.get(line_block_idx, ""):
                start_index = block_wise_indices[line_block_idx]
                final_string = (
                    final_string[: start_index + 1]
                    + [el for el in word_text]
                    + final_string[start_index + len_word + 1 :]
                )
                block_wise_indices[line_block_idx] = start_index + len_word + 1
            else:
                final_string = (
                    final_string[:start_index] + [el for el in word_text] + final_string[start_index + len_word :]
                )
                block_wise_indices[line_block_idx] = start_index + len_word

        text = "".join(final_string).rstrip()
        formatted_line_no = str(line_no).rjust(line_no_length)
        if line_num_at_start:
            texts.append(f"{line_num_start_delim}{formatted_line_no}{line_num_end_delim} " + text)
        else:
            texts.append(text + f" {line_num_start_delim}{line_no}{line_num_end_delim}")

    return "\n".join(texts), df


# ============================================================================


class OCRInputParser:
    """
    Parser component for cleaning and structuring OCR output.

    This parser transforms raw OCR DataFrames into a normalized format that:
    - Removes noise and inconsistencies
    - Groups related content (fields, tables, blocks)
    - Handles multi-page documents
    - Uses layout_conserved_with_lineno for text content
    - Uses AWS Textract for table extraction (HTML output)
    - Produces deterministic, agent-friendly output
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        enable_table_detection: bool = False,
        enable_field_grouping: bool = True,
        combine_words: bool = True,
        gap_factor: float = 0.75,
        aws_region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        """
        Initialize the OCR input parser.

        Args:
            min_confidence: Minimum confidence threshold for OCR words (0.0-1.0)
            enable_table_detection: Whether to detect and extract tables using AWS
            enable_field_grouping: Whether to group fields and key-value pairs
            combine_words: Whether to combine close words into phrases
            gap_factor: Factor for determining word proximity when combining (0.0-1.0)
            aws_region: AWS region for Textract (reads from AWS_REGION env var if None)
            aws_access_key_id: AWS access key ID (reads from AWS_ACCESS_KEY_ID env var if None)
            aws_secret_access_key: AWS secret access key (reads from AWS_SECRET_ACCESS_KEY env var if None)
        """
        self.input_parsing_method = "layout_conserved_with_lineno"  # Fixed method
        self.min_confidence = min_confidence
        self.enable_table_detection = enable_table_detection
        self.enable_field_grouping = enable_field_grouping
        self.combine_words = combine_words
        self.gap_factor = gap_factor
        
        # Read AWS credentials from environment variables if not provided
        self.aws_region = aws_region or os.getenv("AWS_REGION")
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

    def parse(
        self,
        raw_ocr: List[pd.DataFrame],
        page_scalars: Optional[List[Dict[str, Any]]] = None,
        page_images: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Parse raw OCR output into structured format.

        Args:
            raw_ocr: List of DataFrames, one per page, with OCR results
            page_scalars: Optional list of page metadata (width, height, angle)
            page_images: Optional list of page images for AWS table extraction

        Returns:
            Dictionary with structured parsed data:
            {
                "cleaned_ocr": List[pd.DataFrame],  # Cleaned OCR data per page
                "pages": List[Dict],  # Structured page data
                "tables": List[Dict],  # Detected tables (HTML format)
                "fields": Dict,  # Grouped fields/key-value pairs
                "content": str,  # Full document text (layout_conserved_with_lineno)
                "metadata": Dict,  # Parsing metadata
            }
        """
        if not raw_ocr:
            logger.warning("Empty OCR input provided")
            return self._empty_result()

        total_pages = len(raw_ocr)
        logger.info(f"Parsing {total_pages} pages of OCR data using layout_conserved_with_lineno")

        # Step 1: Clean and normalize OCR data
        logger.info("Step 1/5: Cleaning and normalizing OCR data...")
        cleaned_ocr = self._clean_ocr_data(raw_ocr)

        # Step 2: Combine words if enabled
        if self.combine_words:
            logger.info("Step 2/5: Combining words...")
            cleaned_ocr = self._combine_words(cleaned_ocr)

        # Step 3: Extract structured content per page
        logger.info("Step 3/5: Extracting structured content per page...")
        pages_data = []
        all_tables = []
        all_fields = {}
        all_content_parts = []

        # Progress tracking milestones (10%, 25%, 50%, 75%, 100%)
        progress_milestones = {
            int(total_pages * 0.10): "10%",
            int(total_pages * 0.25): "25%",
            int(total_pages * 0.50): "50%",
            int(total_pages * 0.75): "75%",
            total_pages: "100%",
        }
        # Also log every 50 pages for large documents
        log_interval = 50

        for page_idx, ocr_df in enumerate(cleaned_ocr):
            page_number = page_idx + 1
            page_scalar = page_scalars[page_idx] if page_scalars and page_idx < len(page_scalars) else {}
            page_image = page_images[page_idx] if page_images and page_idx < len(page_images) else None

            # Log progress
            if page_number in progress_milestones:
                percentage = progress_milestones[page_number]
                logger.info(f"Progress: {percentage} ({page_number}/{total_pages} pages parsed)")
            elif page_number % log_interval == 0:
                percentage = (page_number / total_pages) * 100
                logger.info(f"Progress: {percentage:.1f}% ({page_number}/{total_pages} pages parsed)")

            # Extract page content using layout_conserved_with_lineno
            page_content = self._extract_page_content(ocr_df)
            all_content_parts.append(page_content)

            # Detect tables using AWS Textract (only if enabled)
            tables = []
            if self.enable_table_detection and page_image is not None:
                tables = self._extract_tables_aws(page_image, page_number, page_scalar)
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
        logger.info("Step 4/5: Generating full document content...")
        full_content = "\n\n".join(all_content_parts)

        # Step 5: Build metadata
        logger.info("Step 5/5: Building metadata...")
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

        logger.info(f"Completed parsing {total_pages} pages. Total words: {metadata['total_words']}, Tables: {metadata['total_tables']}, Fields: {metadata['total_fields']}")
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
        Extract text content from OCR DataFrame using layout_conserved_with_lineno.

        Args:
            ocr_df: OCR DataFrame for a single page

        Returns:
            Extracted text content with line numbers
        """
        if ocr_df is None or ocr_df.empty:
            return ""

        try:
            # Use the local implementation
            text, _ = _get_layout_conserved_text_with_lineno(ocr_df)
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

    def _extract_tables_aws(
        self, page_image: Any, page_number: int, page_scalar: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from page using AWS Textract and convert to HTML.

        Args:
            page_image: Page image (numpy array, PIL Image, or bytes)
            page_number: Page number
            page_scalar: Page dimensions

        Returns:
            List of tables in HTML format
        """
        tables = []

        try:
            # Convert image to bytes if needed
            image_bytes = self._image_to_bytes(page_image)
            if image_bytes is None:
                logger.warning(f"Could not convert page {page_number} image to bytes for AWS table extraction")
                return []

            # Initialize AWS Textract client with credentials from environment or constructor
            client_kwargs = {}
            if self.aws_region:
                client_kwargs["region_name"] = self.aws_region
            if self.aws_access_key_id and self.aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = self.aws_access_key_id
                client_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
            
            textract_client = boto3.client("textract", **client_kwargs)

            # Call AWS Textract with table detection
            response = textract_client.analyze_document(
                Document={"Bytes": image_bytes},
                FeatureTypes=["TABLES"],
            )

            # Parse tables from response
            tables = self._parse_aws_tables_to_html(response, page_number, page_scalar)

        except Exception as e:
            logger.warning(f"Error extracting tables with AWS Textract on page {page_number}: {e}")

        return tables

    def _image_to_bytes(self, image: Any) -> Optional[bytes]:
        """
        Convert various image types to bytes for AWS Textract.

        Args:
            image: Image in various formats (numpy array, PIL Image, bytes, etc.)

        Returns:
            Image bytes or None if conversion fails
        """
        import cv2  # type: ignore
        from PIL import Image
        from io import BytesIO

        try:
            # If already bytes, return as-is
            if isinstance(image, bytes):
                return image

            # Convert numpy array to bytes
            if isinstance(image, np.ndarray):
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # type: ignore[attr-defined]
                success, buffer = cv2.imencode(".jpg", image, encode_params)  # type: ignore[attr-defined]
                if success:
                    return buffer.tobytes()
                return None

            # Convert PIL Image to bytes
            if isinstance(image, Image.Image):
                buffer = BytesIO()
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                image.save(buffer, format="JPEG", quality=85)
                return buffer.getvalue()

            return None

        except Exception as e:
            logger.warning(f"Error converting image to bytes: {e}")
            return None

    def _parse_aws_tables_to_html(
        self, response: Dict[str, Any], page_number: int, page_scalar: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # page_scalar is used in _get_table_bbox
        """
        Parse AWS Textract table response and convert to HTML format.

        Args:
            response: AWS Textract analyze_document response
            page_number: Page number
            page_scalar: Page dimensions

        Returns:
            List of tables with HTML content
        """
        tables = []

        try:
            blocks = response.get("Blocks", [])
            if not blocks:
                return []

            # Build block lookup
            block_by_id = {block["Id"]: block for block in blocks}

            # Find all TABLE blocks
            table_blocks = [block for block in blocks if block["BlockType"] == "TABLE"]

            for table_idx, table_block in enumerate(table_blocks):
                # Get cells in this table
                cell_ids = []
                for rel in table_block.get("Relationships", []):
                    if rel["Type"] == "CHILD":
                        cell_ids.extend(rel["Ids"])

                # Parse cells
                cells = {}
                for cell_id in cell_ids:
                    cell_block = block_by_id.get(cell_id)
                    if cell_block and cell_block["BlockType"] == "CELL":
                        cells[cell_id] = cell_block

                # Build table structure
                html_table = self._build_html_table(cells, block_by_id, page_scalar)

                if html_table:
                    table = {
                        "table_id": f"table_{page_number}_{table_idx}",
                        "page": page_number,
                        "html": html_table,
                        "bbox": self._get_table_bbox(table_block, page_scalar),
                    }
                    tables.append(table)

        except Exception as e:
            logger.warning(f"Error parsing AWS tables to HTML: {e}")

        return tables

    def _build_html_table(
        self, cells: Dict[str, Any], block_by_id: Dict[str, Any], page_scalar: Dict[str, Any]
    ) -> Optional[str]:
        """
        Build HTML table from AWS Textract cells.

        Args:
            cells: Dictionary of cell blocks
            block_by_id: All blocks lookup
            page_scalar: Page dimensions

        Returns:
            HTML table string or None
        """
        try:
            # Group cells by row and column
            rows = {}
            for cell_id, cell_block in cells.items():
                row_index = cell_block.get("RowIndex", 0)
                col_index = cell_block.get("ColumnIndex", 0)

                if row_index not in rows:
                    rows[row_index] = {}

                # Extract cell text
                cell_text = self._extract_cell_text(cell_block, block_by_id)
                rows[row_index][col_index] = cell_text

            if not rows:
                return None

            # Build HTML table
            html_parts = ["<table>"]
            max_cols = max(max(row.keys()) if row else 0 for row in rows.values())

            for row_idx in sorted(rows.keys()):
                html_parts.append("<tr>")
                row = rows[row_idx]

                for col_idx in range(1, max_cols + 1):
                    cell_text = row.get(col_idx, "")
                    # Escape HTML special characters
                    cell_text = (
                        cell_text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace('"', "&quot;")
                        .replace("'", "&#39;")
                    )
                    html_parts.append(f"<td>{cell_text}</td>")

                html_parts.append("</tr>")

            html_parts.append("</table>")
            return "".join(html_parts)

        except Exception as e:
            logger.warning(f"Error building HTML table: {e}")
            return None

    def _extract_cell_text(self, cell_block: Dict[str, Any], block_by_id: Dict[str, Any]) -> str:
        """
        Extract text from a table cell.

        Args:
            cell_block: Cell block from AWS Textract
            block_by_id: All blocks lookup

        Returns:
            Cell text content
        """
        text_parts = []

        # Get word IDs from cell relationships
        for rel in cell_block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for word_id in rel["Ids"]:
                    word_block = block_by_id.get(word_id)
                    if word_block and word_block["BlockType"] == "WORD":
                        text_parts.append(word_block.get("Text", ""))

        return " ".join(text_parts).strip()

    def _get_table_bbox(self, table_block: Dict[str, Any], page_scalar: Dict[str, Any]) -> List[float]:
        """
        Get table bounding box from AWS block.

        Args:
            table_block: Table block from AWS Textract
            page_scalar: Page dimensions

        Returns:
            Bounding box [x0, y0, x2, y2]
        """
        try:
            geometry = table_block.get("Geometry", {})
            bbox = geometry.get("BoundingBox", {})

            page_width = page_scalar.get("width", 1)
            page_height = page_scalar.get("height", 1)

            x0 = bbox.get("Left", 0) * page_width
            y0 = bbox.get("Top", 0) * page_height
            x2 = (bbox.get("Left", 0) + bbox.get("Width", 0)) * page_width
            y2 = (bbox.get("Top", 0) + bbox.get("Height", 0)) * page_height

            return [float(x0), float(y0), float(x2), float(y2)]

        except Exception:
            return [0.0, 0.0, 0.0, 0.0]

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
