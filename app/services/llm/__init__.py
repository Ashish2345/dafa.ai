"""
LLM Service

Handles LLM calls with strict separation of concerns.
Only responsible for calling the LLM API - no business logic.
"""

from .service import LLMService

__all__ = ["LLMService"]
