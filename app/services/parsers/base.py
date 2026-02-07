"""
Abstract base class for document parsers.

All document-specific parsers should inherit from this class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO, Union

from loguru import logger

from app.config.config import ParserConfig
from app.models.schemas import ParsedResponse


class Parser(ABC):
    """
    Abstract base class for all document parsers.

    Implements a 3-step parsing pipeline:
    1. _pre_process: Preprocesses the file (e.g., OCR for scanned documents)
    2. _parse: Parses the preprocessed data using internal algorithms
    3. _post_process: Creates the final ParsedResponse object

    Subclasses must implement all three abstract methods.
    """

    def __init__(self, config: ParserConfig) -> None:
        """
        Initialize the parser with configuration.

        Args:
            config: Parser configuration settings
        """
        self.config = config

    async def parse(self, file_path: Union[str, Path], file_obj: BinaryIO | None = None) -> ParsedResponse:
        """
        Parse a document and return structured content.

        This method orchestrates the 3-step parsing pipeline:
        1. Validates the file
        2. Preprocesses the file (_pre_process)
        3. Parses the preprocessed data (_parse)
        4. Post-processes and creates the response (_post_process)

        Args:
            file_path: Path to the document file
            file_obj: Optional file-like object (for streaming)

        Returns:
            ParsedResponse with extracted content and metadata
        """
        path = Path(file_path)
        logger.info(f"Starting parse pipeline for: {path.name}")

        # Validate file
        await self.validate_file(file_path)

        # Step 1: Pre-process
        logger.debug(f"Step 1: Pre-processing {path.name}")
        preprocessed_data = await self._pre_process(file_path, file_obj)

        # Step 2: Parse
        logger.debug(f"Step 2: Parsing {path.name}")
        parsed_data = await self._parse(preprocessed_data)

        # Step 3: Post-process
        logger.debug(f"Step 3: Post-processing {path.name}")
        result = await self._post_process(parsed_data, file_path)

        logger.info(f"Completed parse pipeline for: {path.name}")
        return result

    @abstractmethod
    async def _pre_process(self, file_path: Union[str, Path], file_obj: BinaryIO | None = None) -> Any:
        """
        Preprocess the file before parsing.

        This step handles file-specific preprocessing such as:
        - Reading the file content
        - Detecting if OCR is needed (for PDFs)
        - Running OCR for scanned/image documents

        Args:
            file_path: Path to the document file
            file_obj: Optional file-like object (for streaming)

        Returns:
            Preprocessed data to be passed to _parse()
        """
        pass

    @abstractmethod
    async def _parse(self, preprocessed_data: Any) -> Any:
        """
        Parse the preprocessed data using internal algorithms.

        This step applies parsing logic to extract structured content
        from the preprocessed data.

        Args:
            preprocessed_data: Data returned from _pre_process()

        Returns:
            Parsed data to be passed to _post_process()
        """
        pass

    @abstractmethod
    async def _post_process(self, parsed_data: Any, file_path: Union[str, Path]) -> ParsedResponse:
        """
        Post-process parsed data and create the final response.

        This step creates the ParsedResponse object with all
        extracted content, metadata, pages, and blocks.

        Args:
            parsed_data: Data returned from _parse()
            file_path: Original file path for metadata extraction

        Returns:
            ParsedResponse with extracted content and metadata
        """
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions this parser supports."""
        pass

    async def validate_file(self, file_path: Union[str, Path]) -> None:
        """
        Validate that file exists and meets size constraints.

        Args:
            file_path: Path to the file to validate

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file exceeds max size or has unsupported extension
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check file size
        file_size = path.stat().st_size
        if file_size > self.config.max_file_size:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed ({self.config.max_file_size} bytes)"
            )

        # Check extension
        extension = path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension '{extension}'. Supported: {self.supported_extensions}")

    def _get_file_metadata(self, file_path: Union[str, Path]) -> dict:
        """
        Extract basic file metadata.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file metadata
        """
        path = Path(file_path)
        stat = path.stat()

        return {
            "filename": path.name,
            "file_size": stat.st_size,
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime,
        }
