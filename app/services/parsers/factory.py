"""
Parser factory for creating and orchestrating document parsers.

Provides a unified interface for parsing different document types.
"""

from pathlib import Path
from typing import Type, Union

from loguru import logger

from app.config.config import (
    DocxParserConfig,
    ExcelParserConfig,
    ImageParserConfig,
    ParserConfig,
    PDFParserConfig,
    RequestConfig,
)
from app.models.enums import FileType
from app.models.schemas import ParsedResponse
from app.services.parsers.base import Parser
from app.services.parsers.docx.parser import DocxParser
from app.services.parsers.excel.parser import ExcelParser
from app.services.parsers.image.parser import ImageParser
from app.services.parsers.pdf.parser import PDFParser
from app.utils.exceptions import ParserNotInitializedError

# Mapping of file extensions to FileType
EXTENSION_TO_FILE_TYPE: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".xlsx": FileType.EXCEL,
    ".xls": FileType.EXCEL,
    ".docx": FileType.DOCX,
    # Image formats
    ".png": FileType.IMAGE,
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".gif": FileType.IMAGE,
    ".tiff": FileType.IMAGE,
    ".tif": FileType.IMAGE,
    ".bmp": FileType.IMAGE,
    ".webp": FileType.IMAGE,
}

# Mapping of FileType to parser class
FILE_TYPE_TO_PARSER: dict[FileType, Type[Parser]] = {
    FileType.PDF: PDFParser,
    FileType.EXCEL: ExcelParser,
    FileType.DOCX: DocxParser,
    FileType.IMAGE: ImageParser,
}

# Mapping of FileType to default config class
FILE_TYPE_TO_CONFIG: dict[FileType, Type[ParserConfig]] = {
    FileType.PDF: PDFParserConfig,
    FileType.EXCEL: ExcelParserConfig,
    FileType.DOCX: DocxParserConfig,
    FileType.IMAGE: ImageParserConfig,
}


class ParserFactory:
    """
    Factory for creating and executing document parsers.

    Provides methods to:
    - Detect file type from extension
    - Create appropriate parser for a file type
    - Parse documents with automatic parser selection
    """

    def __init__(self, request_config: RequestConfig) -> None:
        """
        Initialize the ParserFactory with request configuration.

        Args:
            request_config: Configuration for the parsing request
        """
        self.request_config = request_config
        self.file_path: Path | None = None
        self.file_type: FileType | None = None
        self.parser: Parser | None = None

    @staticmethod
    def detect_file_type(file_path: Union[str, Path]) -> FileType:
        """
        Detect file type from file extension.

        Args:
            file_path: Path to the file

        Returns:
            FileType enum value

        Raises:
            ValueError: If file extension is not supported
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        file_type = EXTENSION_TO_FILE_TYPE.get(extension)
        if file_type is None:
            supported = ", ".join(EXTENSION_TO_FILE_TYPE.keys())
            raise ValueError(f"Unsupported file extension '{extension}'. Supported: {supported}")

        return file_type

    @staticmethod
    def get_parser(
        file_type: FileType,
        config: ParserConfig | None = None,
    ) -> Parser:
        """
        Get a parser instance for the given file type.

        Args:
            file_type: Type of file to parse
            config: Optional parser configuration. If not provided,
                   uses default config for the file type.

        Returns:
            Parser instance configured for the file type

        Raises:
            ValueError: If file type is not supported
        """
        parser_class = FILE_TYPE_TO_PARSER.get(file_type)
        if parser_class is None:
            supported = ", ".join(t.value for t in FILE_TYPE_TO_PARSER.keys())
            raise ValueError(f"No parser available for file type '{file_type.value}'. Supported: {supported}")

        # Use type-specific config if none provided
        if config is None:
            config_class = FILE_TYPE_TO_CONFIG.get(file_type, ParserConfig)
            config = config_class()

        return parser_class(config)

    @staticmethod
    def get_default_config(file_type: FileType) -> ParserConfig:
        """
        Get the default configuration for a file type.

        Args:
            file_type: Type of file

        Returns:
            Default ParserConfig subclass instance for the file type
        """
        config_class = FILE_TYPE_TO_CONFIG.get(file_type, ParserConfig)
        return config_class()

    def _get_config_for_file_type(self, file_type: FileType) -> ParserConfig:
        """
        Get the parser config for a file type from request_config.

        Args:
            file_type: Type of file being parsed

        Returns:
            Appropriate ParserConfig from request_config
        """
        config_mapping = {
            FileType.PDF: self.request_config.pdf_config,
            FileType.EXCEL: self.request_config.excel_config,
            FileType.DOCX: self.request_config.docx_config,
            FileType.IMAGE: self.request_config.image_config,
        }
        return config_mapping.get(file_type, ParserConfig())

    def _pre_process(self, file_path: Union[str, Path]) -> None:
        """
        Pre-process step: Identify file type and initialize the appropriate parser.

        Args:
            file_path: Path to the document file
        """
        self.file_path = Path(file_path)
        logger.info(f"ParserFactory: Pre-processing file '{self.file_path.name}'")

        # Detect file type
        self.file_type = self.detect_file_type(self.file_path)
        logger.debug(f"Detected file type: {self.file_type.value}")

        # Get config for file type from request_config
        config = self._get_config_for_file_type(self.file_type)
        logger.debug(f"Using config: {config.__class__.__name__}")

        # Initialize parser for file type with config
        self.parser = self.get_parser(self.file_type, config)
        logger.debug(f"Initialized parser: {self.parser.__class__.__name__}")

    async def process(self) -> ParsedResponse:
        """
        Process step: Parse the file using the initialized parser.

        Returns:
            ParsedResponse with extracted content and metadata

        Raises:
            ParserNotInitializedError: If _pre_process was not called first
        """
        if self.parser is None or self.file_path is None:
            raise ParserNotInitializedError()

        logger.debug(f"ParserFactory: Processing file '{self.file_path.name}'")
        result = await self.parser.parse(self.file_path)
        return result

    def _post_process(self, result: ParsedResponse) -> ParsedResponse:
        """
        Post-process step: Return the parsed response.

        Args:
            result: The parsed response from process()

        Returns:
            ParsedResponse (unchanged for now)
        """
        if self.file_path is not None:
            logger.info(
                f"ParserFactory: Post-processing complete for '{self.file_path.name}' "
                f"({len(result.pages)} pages, {len(result.content)} chars)"
            )
        return result

    async def parse(self, file_path: Union[str, Path]) -> ParsedResponse:
        """
        Parse a document with automatic file type detection.

        This is the main entry point for parsing documents. It orchestrates
        the 3-step parsing pipeline:
        1. _pre_process: Detects file type and initializes the parser
        2. process: Parses the document using the initialized parser
        3. _post_process: Returns the parsed response

        Args:
            file_path: Path to the document file

        Returns:
            ParsedResponse with extracted content and metadata

        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        # Step 1: Pre-process (detect file type, initialize parser)
        self._pre_process(file_path)

        # Step 2: Process (parse the document)
        result = await self.process()

        # Step 3: Post-process (return the result)
        return self._post_process(result)

    @staticmethod
    def supported_extensions() -> list[str]:
        """
        Get list of all supported file extensions.

        Returns:
            List of supported extensions (e.g., ['.pdf', '.xlsx', ...])
        """
        return list(EXTENSION_TO_FILE_TYPE.keys())

    @staticmethod
    def supported_file_types() -> list[FileType]:
        """
        Get list of all supported file types.

        Returns:
            List of supported FileType enum values
        """
        return list(FILE_TYPE_TO_PARSER.keys())
