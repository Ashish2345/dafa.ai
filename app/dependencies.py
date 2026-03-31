"""
FastAPI dependency injection providers.

Contains functions that provide dependencies to endpoint handlers.
"""

from app.db.mongodb import get_database
from app.services.parsers.factory import ParserFactory


async def get_db():
    """
    Get database instance.

    Returns:
        MongoDB database instance (supports async operations)
    """
    return await get_database()


def get_parser_factory() -> ParserFactory:
    """
    Get parser factory instance.

    Returns:
        ParserFactory for parsing documents
    """
    return ParserFactory()
