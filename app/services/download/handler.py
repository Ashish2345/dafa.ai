"""
File handler for downloading and saving files from various sources.
"""

import tempfile
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import httpx
from fastapi import UploadFile
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from app.settings import settings


def _is_retryable_error(exc: BaseException) -> bool:
    """Check if an exception should trigger a retry.

    Retries on:
    - Transport errors (network issues, connection errors)
    - HTTP 5xx errors (server errors)

    Does not retry on:
    - HTTP 4xx errors (client errors like 404, 403)
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


class FileHandler:
    """
    Handles file acquisition from uploads or URLs.

    Supports:
    - Direct file uploads (UploadFile)
    - URL downloads (signed URLs, public URLs)
    """

    def __init__(
        self,
        timeout: float = 30.0,
        retry_attempts: Optional[int] = None,
        retry_wait_min: Optional[float] = None,
        retry_wait_max: Optional[float] = None,
    ):
        """
        Initialize the file handler.

        Args:
            timeout: HTTP timeout for URL downloads in seconds.
            retry_attempts: Number of retry attempts (defaults to settings).
            retry_wait_min: Minimum wait between retries in seconds (defaults to settings).
            retry_wait_max: Maximum wait between retries in seconds (defaults to settings).
        """
        self.timeout = timeout
        self.retry_attempts = retry_attempts or settings.download_retry_attempts
        self.retry_wait_min = retry_wait_min or settings.download_retry_wait_min
        self.retry_wait_max = retry_wait_max or settings.download_retry_wait_max

    async def process(
        self,
        file_type: Literal["url", "file"],
        file: Optional[UploadFile] = None,
        file_url: Optional[str] = None,
    ) -> str:
        """
        Process file input and return the saved file path.

        Args:
            file_type: Type of input - 'file' for upload, 'url' for URL download.
            file: Uploaded file (required when file_type='file').
            file_url: URL to download from (required when file_type='url').

        Returns:
            Path to the saved temporary file.

        Raises:
            ValueError: If required parameters are missing for the given file_type.
        """
        if file_type == "file":
            if not file:
                raise ValueError("File upload required when file_type='file'")
            return await self.save_upload(file)
        else:
            if not file_url:
                raise ValueError("file_url required when file_type='url'")
            return await self.download_url(file_url)

    async def save_upload(self, file: UploadFile) -> str:
        """
        Save an uploaded file to a temporary location.

        Args:
            file: The uploaded file.

        Returns:
            Path to the saved temporary file.
        """
        suffix = Path(file.filename).suffix if file.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            return tmp.name

    async def download_url(self, url: str) -> str:
        """
        Download a file from a URL and save to a temporary location.

        Includes automatic retry with exponential backoff for transient failures
        (network errors and server 5xx errors).

        Args:
            url: The URL to download from (can be a signed URL).

        Returns:
            Path to the saved temporary file.

        Raises:
            httpx.HTTPError: If the download fails after all retry attempts.
        """
        suffix = self._extract_extension_from_url(url)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(min=self.retry_wait_min, max=self.retry_wait_max),
            retry=retry_if_exception(_is_retryable_error),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url)
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(response.content)
                        return tmp.name

    def _extract_extension_from_url(self, url: str) -> str:
        """
        Extract file extension from URL path.

        Args:
            url: The URL to extract extension from.

        Returns:
            File extension including the dot (e.g., '.pdf'), or empty string.
        """
        parsed = urlparse(url)
        path = parsed.path
        # Remove query params that might be attached
        if "?" in path:
            path = path.split("?")[0]
        return Path(path).suffix
