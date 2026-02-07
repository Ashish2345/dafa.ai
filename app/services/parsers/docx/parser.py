"""
DOCX document parser.

Uses python-docx for parsing Word documents (.docx).
"""

from pathlib import Path
from typing import BinaryIO, Union

from docx import Document
from docx.table import Table
from loguru import logger

from app.config.config import DocxParserConfig, ParserConfig
from app.utils import to_thread
from app.models.enums import ParsingType
from app.models.schemas import Block, Page, ParsedResponse
from app.services.parsers.base import Parser


class DocxParser(Parser):
    """
    Parser for DOCX documents.

    Uses python-docx library to extract text, tables, and metadata
    from Word documents.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        """
        Initialize the DOCX parser.

        Args:
            config: Parser configuration. If not provided, uses default DocxParserConfig.
        """
        if config is None:
            config = DocxParserConfig()
        elif not isinstance(config, DocxParserConfig):
            config = DocxParserConfig(**config.model_dump())
        super().__init__(config)

    @property
    def docx_config(self) -> DocxParserConfig:
        """Get the DOCX-specific configuration."""
        return self.config  # type: ignore

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return [".docx"]

    async def parse(self, file_path: Union[str, Path], file_obj: BinaryIO | None = None) -> ParsedResponse:
        """
        Parse a DOCX document and return structured content.

        Args:
            file_path: Path to the DOCX file
            file_obj: Optional file-like object (not used, path required)

        Returns:
            ParsedResponse with extracted content and metadata
        """
        await self.validate_file(file_path)

        path = Path(file_path)
        logger.info(f"Parsing DOCX: {path.name}")

        # Run DOCX parsing in thread pool to avoid blocking
        result = await to_thread(self._parse_sync, path)

        return result

    def _parse_sync(self, file_path: Path) -> ParsedResponse:
        """
        Synchronous DOCX parsing implementation.

        Args:
            file_path: Path to the DOCX file

        Returns:
            ParsedResponse with extracted content
        """
        document = Document(str(file_path))

        # Extract all content
        content_parts: list[str] = []
        blocks: list[Block] = []
        block_idx = 0

        # Process paragraphs and tables in document order
        for element in document.element.body:
            if element.tag.endswith("p"):  # Paragraph
                # Find the corresponding paragraph object
                for para in document.paragraphs:
                    if para._element is element:
                        text = para.text.strip()
                        if text:
                            content_parts.append(text)

                            block_data = {
                                "block_id": f"block_1_{block_idx}",
                                "block_parsed_content": text,
                            }

                            # Add style info if configured
                            if self.docx_config.extract_styles and para.style:
                                block_data["block_html_tags"] = {
                                    "style_name": para.style.name,
                                }

                            blocks.append(Block(**block_data))
                            block_idx += 1
                        break

            elif element.tag.endswith("tbl"):  # Table
                # Find the corresponding table object
                for table in document.tables:
                    if table._element is element:
                        table_text = self._extract_table_text(table)
                        if table_text:
                            content_parts.append(table_text)

                            blocks.append(
                                Block(
                                    block_id=f"block_1_{block_idx}",
                                    block_parsed_content=table_text,
                                    block_html_tags={"type": "table"},
                                )
                            )
                            block_idx += 1
                        break

        # Extract comments if configured
        comments_text = ""
        if self.docx_config.extract_comments:
            comments_text = self._extract_comments(document)
            if comments_text:
                content_parts.append(f"\n--- Comments ---\n{comments_text}")

        # Create single page (DOCX doesn't have physical pages)
        full_content = "\n\n".join(content_parts)

        page = Page(
            page_number=1,
            page_content=full_content,
            blocks=blocks,
        )

        # Gather metadata
        file_metadata = self._get_file_metadata(file_path)
        if self.docx_config.extract_metadata:
            file_metadata.update(self._extract_document_metadata(document))

        return ParsedResponse(
            content=full_content,
            file_type="docx",
            file_metadata=file_metadata,
            parsing_type=ParsingType.TEXT,
            pages=[page],
        )

    def _extract_table_text(self, table: Table) -> str:
        """
        Extract text content from a DOCX table.

        Args:
            table: python-docx Table object

        Returns:
            Table content as formatted text
        """
        rows_text = []

        for row in table.rows:
            cells_text = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                cells_text.append(cell_text)
            rows_text.append("\t".join(cells_text))

        return "\n".join(rows_text)

    def _extract_comments(self, document: Document) -> str:
        """
        Extract comments from the document.

        Args:
            document: python-docx Document object

        Returns:
            Comments as formatted text
        """
        # python-docx doesn't have direct comment support
        # This would require parsing the XML directly
        # For now, return empty string
        return ""

    def _extract_document_metadata(self, document: Document) -> dict:
        """
        Extract document metadata (author, title, etc.).

        Args:
            document: python-docx Document object

        Returns:
            Dictionary with document metadata
        """
        metadata = {}
        core_props = document.core_properties

        if core_props.author:
            metadata["author"] = core_props.author
        if core_props.title:
            metadata["title"] = core_props.title
        if core_props.subject:
            metadata["subject"] = core_props.subject
        if core_props.created:
            metadata["created"] = core_props.created.isoformat()
        if core_props.modified:
            metadata["modified"] = core_props.modified.isoformat()
        if core_props.last_modified_by:
            metadata["last_modified_by"] = core_props.last_modified_by

        # Count elements
        metadata["paragraph_count"] = len(document.paragraphs)
        metadata["table_count"] = len(document.tables)

        return metadata
