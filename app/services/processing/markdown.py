"""
Document Processor

Converts OCR output to clean Markdown format for RAG ingestion.
This is the bridge between the OCR parser and the chunking system.
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.parsers.ocr_input_parser import OCRInputParser


PAGE_SEPARATOR = "\n\n---\n\n"


class DocumentProcessor:
    """
    Processes OCR output and converts it to Markdown format.

    This class takes the structured output from OCRInputParser and converts it
    to clean Markdown that can be used by the RAG chunking system.
    """

    def __init__(self):
        """Initialize the document processor."""
        self.ocr_parser = OCRInputParser(
            min_confidence=0.5,
            enable_table_detection=True,
            enable_field_grouping=True,
            combine_words=True,
        )

    def process_to_markdown(
        self,
        raw_ocr: List,
        page_scalars: Optional[List[Dict[str, Any]]] = None,
        page_images: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process OCR data and convert to Markdown format.

        Args:
            raw_ocr: List of OCR DataFrames (one per page)
            page_scalars: Optional page metadata (width, height, angle)
            page_images: Optional page images for table extraction

        Returns:
            Dictionary with Markdown content and metadata:
            {
                "markdown": str,  # Complete document in Markdown format
                "pages": List[Dict],  # Per-page Markdown with metadata
                "tables": List[Dict],  # Table information
                "metadata": Dict,  # Document metadata
                "page_bbox_map": List[Dict],  # Character-offset to page/bbox mapping
            }
        """
        logger.info("Processing document to Markdown format")

        # Parse OCR data
        parsed_result = self.ocr_parser.parse(
            raw_ocr=raw_ocr,
            page_scalars=page_scalars,
            page_images=page_images,
        )

        # Single parse: build all output from _convert_to_markdown_with_bboxes
        full_markdown, page_bbox_map = self._convert_to_markdown_with_bboxes(parsed_result)
        word_bboxes = self._extract_word_bboxes(parsed_result, full_markdown, page_bbox_map)

        pages = parsed_result.get("pages", [])
        tables = parsed_result.get("tables", [])
        fields = parsed_result.get("fields", {})
        metadata = parsed_result.get("metadata", {})

        # Reconstruct per-page markdown entries (reuses already-computed page md)
        markdown_pages = []
        for page_data in pages:
            page_number = page_data.get("page_number", 0)
            page_content = page_data.get("content", "")
            page_blocks = page_data.get("blocks", [])
            page_tables = page_data.get("tables", [])
            page_markdown = self._convert_page_to_markdown(
                page_content, page_blocks, page_tables, page_number
            )
            markdown_pages.append({
                "page_number": page_number,
                "markdown": page_markdown,
                "word_count": page_data.get("word_count", 0),
                "bbox": page_data.get("bbox", [0, 0, 0, 0]),
            })

        formatted_tables = []
        for table in tables:
            formatted_tables.append({
                "table_id": table.get("table_id", ""),
                "page": table.get("page", 0),
                "html": table.get("html", ""),
                "markdown": self._html_table_to_markdown(table.get("html", "")),
                "bbox": table.get("bbox", [0, 0, 0, 0]),
            })

        return {
            "markdown": full_markdown,
            "pages": markdown_pages,
            "tables": formatted_tables,
            "fields": fields,
            "metadata": metadata,
            "page_bbox_map": page_bbox_map,
            "word_bboxes": word_bboxes,
        }

    def _convert_to_markdown(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert parsed OCR result to Markdown format.

        Args:
            parsed_result: Result from OCRInputParser.parse()

        Returns:
            Markdown-formatted document
        """
        pages = parsed_result.get("pages", [])
        tables = parsed_result.get("tables", [])
        fields = parsed_result.get("fields", {})
        metadata = parsed_result.get("metadata", {})

        # Convert each page to Markdown
        markdown_pages = []
        all_markdown_parts = []

        for page_data in pages:
            page_number = page_data.get("page_number", 0)
            page_content = page_data.get("content", "")  # layout_conserved_with_lineno format
            page_blocks = page_data.get("blocks", [])
            page_tables = page_data.get("tables", [])

            # Convert page content to Markdown
            page_markdown = self._convert_page_to_markdown(
                page_content, page_blocks, page_tables, page_number
            )

            markdown_pages.append({
                "page_number": page_number,
                "markdown": page_markdown,
                "word_count": page_data.get("word_count", 0),
                "bbox": page_data.get("bbox", [0, 0, 0, 0]),
            })

            all_markdown_parts.append(page_markdown)

        # Combine all pages
        full_markdown = PAGE_SEPARATOR.join(all_markdown_parts)

        # Format tables metadata
        formatted_tables = []
        for table in tables:
            formatted_tables.append({
                "table_id": table.get("table_id", ""),
                "page": table.get("page", 0),
                "html": table.get("html", ""),
                "markdown": self._html_table_to_markdown(table.get("html", "")),
                "bbox": table.get("bbox", [0, 0, 0, 0]),
            })

        return {
            "markdown": full_markdown,
            "pages": markdown_pages,
            "tables": formatted_tables,
            "fields": fields,
            "metadata": metadata,
        }

    def _convert_to_markdown_with_bboxes(
        self, parsed_result: Dict[str, Any]
    ) -> tuple[str, list[dict]]:
        """
        Convert parsed OCR result to Markdown and produce a page_bbox_map.

        The page_bbox_map records the character-offset range in the final
        markdown string that corresponds to each page, together with that
        page's bounding box.  Offsets are computed to match exactly the
        output produced by _convert_to_markdown() (pages joined with
        "\\n\\n---\\n\\n").

        Args:
            parsed_result: Result from OCRInputParser.parse()

        Returns:
            tuple of (markdown_string, page_bbox_map) where page_bbox_map is:
            [
                {
                    "start_char": int,
                    "end_char": int,
                    "page": int,
                    "bbox": {"x0": float, "y0": float, "x2": float, "y2": float},
                },
                ...
            ]
        """
        pages = parsed_result.get("pages", [])
        page_markdown_parts: List[str] = []
        page_bbox_map: List[Dict[str, Any]] = []

        for page_data in pages:
            page_number = page_data.get("page_number", 0)
            page_content = page_data.get("content", "")
            page_blocks = page_data.get("blocks", [])
            page_tables = page_data.get("tables", [])
            raw_bbox = page_data.get("bbox", None)

            page_md = self._convert_page_to_markdown(
                page_content, page_blocks, page_tables, page_number
            )
            page_markdown_parts.append(page_md)

            # Only record a mapping entry when there is actual content and a
            # valid bbox.  raw_bbox arrives as a list [x0, y0, x2, y2].
            if raw_bbox is not None and page_md.strip():
                # Compute start_char as the length of everything assembled so
                # far: all previous page strings plus their separators.
                preceding = PAGE_SEPARATOR.join(page_markdown_parts[:-1])
                start_char = len(preceding) + (len(PAGE_SEPARATOR) if len(page_markdown_parts) > 1 else 0)
                end_char = start_char + len(page_md)

                # Normalise bbox regardless of whether it arrived as a list or
                # a dict (the OCR parser currently returns a list).
                if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
                    x0, y0, x2, y2 = (float(v) for v in raw_bbox[:4])
                elif isinstance(raw_bbox, dict):
                    x0 = float(raw_bbox.get("x0", 0))
                    y0 = float(raw_bbox.get("y0", 0))
                    x2 = float(raw_bbox.get("x2", 0))
                    y2 = float(raw_bbox.get("y2", 0))
                else:
                    x0 = y0 = x2 = y2 = 0.0

                page_bbox_map.append({
                    "start_char": start_char,
                    "end_char": end_char,
                    "page": page_number,
                    "bbox": {"x0": x0, "y0": y0, "x2": x2, "y2": y2},
                })

        full_markdown = PAGE_SEPARATOR.join(page_markdown_parts)
        return full_markdown, page_bbox_map

    def _extract_word_bboxes(
        self,
        parsed_result: dict,
        full_markdown: str,
        page_bbox_map: list[dict],
    ) -> list[dict]:
        """Extract word-level bounding boxes with character offsets.

        Uses the cleaned OCR DataFrames (which retain word-level x0/y0/x2/y2)
        and maps each word to its approximate character offset in the full
        markdown string using the page_bbox_map for page-level start_char.

        Returns:
            List of dicts, one per page:
            [{"page": 1, "words": [{"text": ..., "x0": ..., "char_offset": ...}, ...]}, ...]
        """
        cleaned_ocr = parsed_result.get("cleaned_ocr", [])
        pages_data = parsed_result.get("pages", [])
        result = []

        for page_idx, ocr_df in enumerate(cleaned_ocr):
            if ocr_df.empty:
                continue

            page_number = pages_data[page_idx]["page_number"] if page_idx < len(pages_data) else page_idx + 1

            # Find this page's start_char from page_bbox_map
            page_start_char = 0
            for entry in page_bbox_map:
                if entry["page"] == page_number:
                    page_start_char = entry["start_char"]
                    break

            # Find this page's end_char for extracting the page markdown slice
            page_end_char = len(full_markdown)
            for entry in page_bbox_map:
                if entry["page"] == page_number:
                    page_end_char = entry["end_char"]
                    break
            page_md = full_markdown[page_start_char:page_end_char]

            # Build word list — find each word sequentially in the page markdown
            # for accurate char_offsets instead of a running counter.
            # Uses forward-only cursor but tries nearby matches for duplicate words.
            words = []
            search_cursor = 0  # position within page_md
            page_md_lower = page_md.lower()
            for _, row in ocr_df.iterrows():
                text = str(row.get("Text", "")).strip()
                if not text:
                    continue

                x0 = float(row["x0"])
                y0 = float(row["y0"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])

                # Validate coordinates are normalized 0-1
                if x2 > 1.5 or y2 > 1.5:
                    # Likely pixel coords — skip (shouldn't happen with proper OCR)
                    logger.warning(f"Word '{text}' has non-normalized coords x2={x2}, y2={y2}, skipping")
                    continue

                # Search forward from cursor for this word
                pos = page_md.find(text, search_cursor)
                if pos == -1:
                    # Case-insensitive fallback
                    pos = page_md_lower.find(text.lower(), search_cursor)
                if pos >= 0:
                    char_offset = page_start_char + pos
                    search_cursor = pos + len(text)
                else:
                    # Word not found — estimate from cursor position
                    char_offset = page_start_char + search_cursor

                words.append({
                    "text": text,
                    "x0": x0,
                    "y0": y0,
                    "x2": x2,
                    "y2": y2,
                    "block": int(row.get("block", 0)),
                    "line": int(row.get("line", 0)),
                    "char_offset": char_offset,
                })

            if words:
                result.append({"page": page_number, "words": words})

        return result

    def _convert_page_to_markdown(
        self,
        page_content: str,
        blocks: List[Dict],  # Reserved for future use
        tables: List[Dict],
        page_number: int,
    ) -> str:
        """
        Convert a single page's content to Markdown.

        Args:
            page_content: Text content with line numbers (layout_conserved_with_lineno)
            blocks: List of block dictionaries
            tables: List of table dictionaries for this page
            page_number: Page number

        Returns:
            Markdown string for the page
        """
        markdown_parts = []

        # Add page header
        markdown_parts.append(f"## Page {page_number}\n")

        # Convert content (remove line number markers for cleaner Markdown)
        # Keep line numbers in comments for reference
        clean_content = self._remove_line_number_markers(page_content)
        if clean_content.strip():
            markdown_parts.append(clean_content)
            markdown_parts.append("")  # Empty line

        # Add tables
        for table in tables:
            table_html = table.get("html", "")
            if table_html:
                table_markdown = self._html_table_to_markdown(table_html)
                markdown_parts.append(table_markdown)
                markdown_parts.append("")  # Empty line

        return "\n".join(markdown_parts)

    def _remove_line_number_markers(self, text: str) -> str:
        """
        Remove line number markers (-><num><-) from text while preserving content.

        Args:
            text: Text with line number markers

        Returns:
            Clean text without markers
        """
        # Pattern: -><number><- or -><number><- at start of line
        pattern = r"->\d+<-\s*"
        clean_text = re.sub(pattern, "", text)
        return clean_text.strip()

    def _html_table_to_markdown(self, html: str) -> str:
        """
        Convert HTML table to Markdown table format.

        Args:
            html: HTML table string

        Returns:
            Markdown table string
        """
        if not html:
            return ""

        from html.parser import HTMLParser

        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows = []
                self.current_row = []
                self.in_cell = False
                self.current_cell = ""

            def handle_starttag(self, tag, attrs):
                if tag == "tr":
                    self.current_row = []
                elif tag == "td":
                    self.in_cell = True
                    self.current_cell = ""

            def handle_endtag(self, tag):
                if tag == "td":
                    self.in_cell = False
                    self.current_row.append(self.current_cell.strip())
                elif tag == "tr":
                    if self.current_row:
                        self.rows.append(self.current_row)

            def handle_data(self, data):
                if self.in_cell:
                    self.current_cell += data

        try:
            parser = TableParser()
            parser.feed(html)

            if not parser.rows:
                return ""

            # Convert to Markdown table
            markdown_lines = []
            for i, row in enumerate(parser.rows):
                # Escape pipes in cell content
                escaped_row = [cell.replace("|", "\\|") for cell in row]
                markdown_lines.append("| " + " | ".join(escaped_row) + " |")

                # Add header separator after first row
                if i == 0:
                    markdown_lines.append("| " + " | ".join(["---"] * len(row)) + " |")

            return "\n".join(markdown_lines)

        except (ValueError, AttributeError) as e:
            logger.warning(f"Error converting HTML table to Markdown: {e}")
            # Fallback: extract plain text
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return f"```\n{text}\n```"
