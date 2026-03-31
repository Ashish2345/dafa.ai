"""
Custom exceptions for the document parser application.

All custom exceptions inherit from dafaaiError for easy catching
and handling throughout the application.
"""

from typing import Any, Dict, Optional

from loguru import logger


class AppException(Exception):
    """Base exception class for the application"""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

        logger.error(f"[{self.status_code}] {self.error_code}: {self.message} | Details: {self.details}")

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, error_code={self.error_code!r})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
            }
        }


class dafaaiError(AppException):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class ValidationError(dafaaiError):
    """Raised when file validation fails."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class ParsingError(dafaaiError):
    """Raised when document parsing fails."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class UnsupportedFileTypeError(dafaaiError):
    """Raised when file type is not supported."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class StorageError(dafaaiError):
    """Raised when file storage operations fail."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 401,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class DatabaseError(dafaaiError):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str,
        error_code: str = "E_DATABASE_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class OCRError(dafaaiError):
    """Raised when OCR processing fails."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class AuthenticationError(dafaaiError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class ConfigurationError(dafaaiError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class FileDetectionError(dafaaiError):
    """Raised when file type cannot be determined."""

    def __init__(
        self,
        message: str,
        error_code: str = "E_FILE_DETECTION_ERROR",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class AlgorithmNotFoundError(dafaaiError):
    """Raised when a parsing algorithm is not found."""

    def __init__(
        self,
        algorithm_name: str,
        available_algorithms: Optional[list] = None,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ):
        message = f"Parsing algorithm not found: {algorithm_name}"
        if available_algorithms:
            message += f". Available: {', '.join(available_algorithms)}"

        super().__init__(
            message=message,
            error_code="E_ALGORITHM_NOT_FOUND",
            status_code=status_code,
            details=details or {"algorithm": algorithm_name, "available": available_algorithms},
        )


class ParserNotInitializedError(dafaaiError):
    """Raised when parser operations are attempted before initialization."""

    def __init__(
        self,
        message: str = "Parser not initialized. Ensure the parser is properly configured before processing documents.",
        error_code: str = "E_PARSER_NOT_INITIALIZED",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code, status_code, details)
