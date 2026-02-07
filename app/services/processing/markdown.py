"""
Document Processor

Converts OCR output to clean Markdown format for RAG ingestion.
This is the bridge between the OCR parser and the chunking system.
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.parsers.ocr_input_parser import OCRInputParser


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
            }
        """
        logger.info("Processing document to Markdown format")

        # Parse OCR data
        parsed_result = self.ocr_parser.parse(
            raw_ocr=raw_ocr,
            page_scalars=page_scalars,
            page_images=page_images,
        )

        # Convert to Markdown
        markdown_result = self._convert_to_markdown(parsed_result)

        return markdown_result

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
        full_markdown = "\n\n---\n\n".join(all_markdown_parts)

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
