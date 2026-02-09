"""
Security utilities for API key validation and other security functions.
"""

import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request, status

from app.settings import settings
from app.utils.exceptions import AuthenticationError


def generate_api_key(length: int = 32) -> str:
    """
    Generate a secure random API key.

    Args:
        length: Length of the API key

    Returns:
        Randomly generated API key
    """
    return secrets.token_urlsafe(length)


def validate_api_key_value(api_key: str) -> bool:
    """
    Validate if an API key is in the list of valid keys.

    Args:
        api_key: API key to validate

    Returns:
        True if valid, False otherwise
    """
    return api_key in settings.api_keys


async def get_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias=settings.api_key_header)
) -> str:
    """
    Dependency to extract and validate API key from request headers.

    Skips validation for OPTIONS requests (CORS preflight).

    Args:
        request: FastAPI request object
        x_api_key: API key from request header

    Returns:
        Validated API key (or empty string for OPTIONS requests)

    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Skip API key validation for OPTIONS requests (CORS preflight)
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not validate_api_key_value(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key


def verify_api_key(api_key: str) -> None:
    """
    Verify API key, raising exception if invalid.

    Args:
        api_key: API key to verify

    Raises:
        AuthenticationError: If API key is invalid
    """
    if not validate_api_key_value(api_key):
        raise AuthenticationError("Invalid API key")
