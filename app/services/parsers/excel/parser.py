"""
Excel document parser.

Uses openpyxl and pandas for parsing Excel documents (.xlsx, .xls).
"""

from pathlib import Path
from typing import Any, BinaryIO, Union

import pandas as pd
from loguru import logger
from openpyxl import load_workbook

from app.config.config import ExcelParserConfig, ParserConfig
from app.utils import to_thread
from app.models.enums import ParsingType
from app.models.schemas import Block, Page, ParsedResponse
from app.services.parsers.base import Parser


class ExcelParser(Parser):
    """
    Parser for Excel documents (.xlsx, .xls).

    Uses openpyxl for .xlsx files and pandas with xlrd for .xls files.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        """
        Initialize the Excel parser.

        Args:
            config: Parser configuration. If not provided, uses default ExcelParserConfig.
        """
        if config is None:
            config = ExcelParserConfig()
        elif not isinstance(config, ExcelParserConfig):
            config = ExcelParserConfig(**config.model_dump())
        super().__init__(config)

    @property
    def excel_config(self) -> ExcelParserConfig:
        """Get the Excel-specific configuration."""
        return self.config  # type: ignore

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return [".xlsx", ".xls"]

    async def parse(self, file_path: Union[str, Path], file_obj: BinaryIO | None = None) -> ParsedResponse:
        """
        Parse an Excel document and return structured content.

        Args:
            file_path: Path to the Excel file
            file_obj: Optional file-like object (not used, path required)

        Returns:
            ParsedResponse with extracted content and metadata
        """
        await self.validate_file(file_path)

        path = Path(file_path)
        logger.info(f"Parsing Excel: {path.name}")

        # Run Excel parsing in thread pool to avoid blocking
        result = await to_thread(self._parse_sync, path)

        return result

    def _parse_sync(self, file_path: Path) -> ParsedResponse:
        """
        Synchronous Excel parsing implementation.

        Args:
            file_path: Path to the Excel file

        Returns:
            ParsedResponse with extracted content
        """
        extension = file_path.suffix.lower()

        if extension == ".xlsx":
            return self._parse_xlsx(file_path)
        else:
            return self._parse_xls(file_path)

    def _parse_xlsx(self, file_path: Path) -> ParsedResponse:
        """
        Parse .xlsx files using openpyxl.

        Args:
            file_path: Path to the Excel file

        Returns:
            ParsedResponse with extracted content
        """
        pages: list[Page] = []
        all_content: list[str] = []

        workbook = load_workbook(str(file_path), read_only=True, data_only=True)

        try:
            sheet_names = self._get_sheets_to_parse(workbook.sheetnames)

            for sheet_idx, sheet_name in enumerate(sheet_names, start=1):
                sheet = workbook[sheet_name]

                # Extract content from sheet
                sheet_content, blocks = self._extract_sheet_content(sheet, sheet_idx)
                all_content.append(f"=== {sheet_name} ===\n{sheet_content}")

                page = Page(
                    page_number=sheet_idx,
                    page_name=sheet_name,
                    page_content=sheet_content,
                    blocks=blocks,
                )
                pages.append(page)
        finally:
            workbook.close()

        # Gather metadata
        file_metadata = self._get_file_metadata(file_path)
        file_metadata["sheet_count"] = len(pages)
        file_metadata["sheet_names"] = [p.page_name for p in pages]

        full_content = "\n\n".join(all_content)

        return ParsedResponse(
            content=full_content,
            file_type="xlsx",
            file_metadata=file_metadata,
            parsing_type=ParsingType.STRUCTURED,
            pages=pages,
        )

    def _parse_xls(self, file_path: Path) -> ParsedResponse:
        """
        Parse .xls files using pandas with xlrd.

        Args:
            file_path: Path to the Excel file

        Returns:
            ParsedResponse with extracted content
        """
        pages: list[Page] = []
        all_content: list[str] = []

        # Read all sheets
        excel_file = pd.ExcelFile(str(file_path), engine="xlrd")

        try:
            sheet_names = self._get_sheets_to_parse(excel_file.sheet_names)

            for sheet_idx, sheet_name in enumerate(sheet_names, start=1):
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

                # Convert to text content
                sheet_content = df.to_string(index=False, header=False)
                all_content.append(f"=== {sheet_name} ===\n{sheet_content}")

                # Create blocks for each row
                blocks = self._create_blocks_from_df(df, sheet_idx)

                page = Page(
                    page_number=sheet_idx,
                    page_name=sheet_name,
                    page_content=sheet_content,
                    blocks=blocks,
                )
                pages.append(page)
        finally:
            excel_file.close()

        # Gather metadata
        file_metadata = self._get_file_metadata(file_path)
        file_metadata["sheet_count"] = len(pages)
        file_metadata["sheet_names"] = [p.page_name for p in pages]

        full_content = "\n\n".join(all_content)

        return ParsedResponse(
            content=full_content,
            file_type="xlsx",
            file_metadata=file_metadata,
            parsing_type=ParsingType.STRUCTURED,
            pages=pages,
        )

    def _get_sheets_to_parse(self, available_sheets: list[str]) -> list[str]:
        """
        Determine which sheets to parse based on config.

        Args:
            available_sheets: List of all sheet names in the workbook

        Returns:
            List of sheet names to parse
        """
        if self.excel_config.sheet_names:
            # Filter to only requested sheets that exist
            return [s for s in self.excel_config.sheet_names if s in available_sheets]
        return available_sheets

    def _extract_sheet_content(self, sheet, sheet_idx: int) -> tuple[str, list[Block]]:
        """
        Extract content from an openpyxl worksheet.

        Args:
            sheet: openpyxl worksheet
            sheet_idx: Sheet index for block IDs

        Returns:
            Tuple of (text content, list of blocks)
        """
        rows_content: list[str] = []
        blocks: list[Block] = []

        for row_idx, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            # Convert row values to strings
            row_values = [self._cell_to_string(cell) for cell in row]

            # Skip empty rows
            if not any(row_values):
                continue

            row_text = "\t".join(row_values)
            rows_content.append(row_text)

            # Create a block for each non-empty row
            blocks.append(
                Block(
                    block_id=f"block_{sheet_idx}_{row_idx}",
                    block_parsed_content=row_text,
                    block_cells={
                        "start_row": row_idx,
                        "end_row": row_idx,
                        "start_col": 1,
                        "end_col": len(row),
                    },
                )
            )

        return "\n".join(rows_content), blocks

    def _create_blocks_from_df(self, df: pd.DataFrame, sheet_idx: int) -> list[Block]:
        """
        Create blocks from a pandas DataFrame.

        Args:
            df: DataFrame with sheet data
            sheet_idx: Sheet index for block IDs

        Returns:
            List of Block objects
        """
        blocks = []

        for row_idx, row in df.iterrows():
            row_values = [self._cell_to_string(cell) for cell in row.values]

            if not any(row_values):
                continue

            row_text = "\t".join(row_values)

            blocks.append(
                Block(
                    block_id=f"block_{sheet_idx}_{row_idx + 1}",
                    block_parsed_content=row_text,
                    block_cells={
                        "start_row": row_idx + 1,
                        "end_row": row_idx + 1,
                        "start_col": 1,
                        "end_col": len(row),
                    },
                )
            )

        return blocks

    def _cell_to_string(self, cell: Any) -> str:
        """
        Convert a cell value to string.

        Args:
            cell: Cell value (any type)

        Returns:
            String representation
        """
        if cell is None:
            return ""

        if isinstance(cell, (pd.Timestamp, pd.Timestamp)):
            return cell.strftime(self.excel_config.date_format)

        return str(cell)
