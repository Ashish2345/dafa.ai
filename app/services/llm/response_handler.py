"""
Response Handler for LLM Service

Handles response processing, validation, sanitization, and formatting.
"""

import re
from typing import Dict, Optional, Tuple

from loguru import logger


class ResponseHandler:
    """
    Handles LLM response processing and validation.
    
    Provides professional response formatting, validation, and edge case handling.
    """

    @staticmethod
    def validate_and_sanitize(
        text: str,
        finish_reason: Optional[str] = None,
        is_truncated: bool = False,
    ) -> Tuple[str, Dict[str, any]]:
        """
        Validate and sanitize LLM response.
        
        Args:
            text: Raw response text
            finish_reason: Finish reason from API (e.g., "MAX_TOKENS", "STOP")
            is_truncated: Whether response was truncated
            
        Returns:
            Tuple of (sanitized_text, metadata_dict)
        """
        if not text:
            return "", {"is_empty": True, "is_complete": False}
        
        # Sanitize text
        sanitized = ResponseHandler._sanitize_text(text)
        
        # Check completeness
        is_complete = ResponseHandler._check_completeness(sanitized, finish_reason, is_truncated)
        
        # Extract metadata
        metadata = {
            "is_empty": False,
            "is_complete": is_complete,
            "is_truncated": is_truncated or (finish_reason == "MAX_TOKENS"),
            "finish_reason": finish_reason,
            "length": len(sanitized),
            "word_count": len(sanitized.split()),
            "has_citations": ResponseHandler._has_citations(sanitized),
            "has_formatting": ResponseHandler._has_formatting(sanitized),
        }
        
        return sanitized, metadata

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """
        Sanitize and clean response text.
        
        Args:
            text: Raw text
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix common formatting issues
        text = re.sub(r'\.{3,}', '...', text)  # Multiple dots
        text = re.sub(r' {2,}', ' ', text)  # Multiple spaces
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text

    @staticmethod
    def _check_completeness(
        text: str,
        finish_reason: Optional[str] = None,
        is_truncated: bool = False,
    ) -> bool:
        """
        Check if response appears complete.
        
        Args:
            text: Response text
            finish_reason: Finish reason from API
            is_truncated: Whether response was truncated
            
        Returns:
            True if response appears complete
        """
        if not text:
            return False
        
        if is_truncated or finish_reason == "MAX_TOKENS":
            return False
        
        if finish_reason == "STOP":
            return True
        
        # Check if text ends properly
        text_stripped = text.rstrip()
        proper_endings = (".", "!", "?", ":", ")", "}", "]")
        
        # Check last 50 characters for proper ending
        last_chars = text_stripped[-50:]
        if any(last_chars.endswith(ending) for ending in proper_endings):
            return True
        
        # Check if it ends with a complete sentence pattern
        if re.search(r'[.!?]\s*$', text_stripped):
            return True
        
        # If it's very short, might be complete even without proper ending
        if len(text_stripped) < 50:
            return True
        
        return False

    @staticmethod
    def _has_citations(text: str) -> bool:
        """Check if response contains citations (Act names, Section numbers, etc.)."""
        citation_patterns = [
            r'Section\s+\d+',
            r'Act\s+\d+',
            r'Chapter\s+\d+',
            r'Article\s+\d+',
            r'\[.*?Act.*?\]',
            r'\(.*?Section.*?\)',
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False

    @staticmethod
    def _has_formatting(text: str) -> bool:
        """Check if response has formatting (lists, headers, etc.)."""
        formatting_indicators = [
            r'^\d+\.\s',  # Numbered list
            r'^[-*]\s',  # Bullet list
            r'^#{1,6}\s',  # Markdown headers
            r'\*\*.*?\*\*',  # Bold
            r'_.*?_',  # Italic
        ]
        
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            for pattern in formatting_indicators:
                if re.search(pattern, line):
                    return True
        
        return False

    @staticmethod
    def format_response(
        text: str,
        metadata: Dict[str, any],
        add_warning: bool = True,
    ) -> str:
        """
        Format response with warnings if needed.
        
        Args:
            text: Response text
            metadata: Response metadata
            add_warning: Whether to add truncation warning
            
        Returns:
            Formatted response text
        """
        if not text:
            return "I apologize, but I couldn't generate a response. Please try again."
        
        formatted = text
        
        # Add warning if truncated
        if add_warning and metadata.get("is_truncated"):
            warning = (
                "\n\n---\n"
                "⚠️ *Note: This response was truncated due to length limits. "
                "The information provided may be incomplete. Please refine your query for more specific information.*"
            )
            formatted = text + warning
        
        # Add note if incomplete
        if add_warning and not metadata.get("is_complete") and not metadata.get("is_truncated"):
            # Check if it ends mid-sentence
            if not text.rstrip().endswith((".", "!", "?", ":", ")", "}", "]")):
                note = (
                    "\n\n---\n"
                    "ℹ️ *Note: This response may be incomplete. "
                    "If you need more information, please ask a follow-up question.*"
                )
                formatted = text + note
        
        return formatted

    @staticmethod
    def create_error_response(
        error: Exception,
        error_type: str = "unknown",
        retry_suggested: bool = False,
    ) -> Dict[str, any]:
        """
        Create a professional error response.
        
        Args:
            error: Exception that occurred
            error_type: Type of error (e.g., "api_error", "timeout", "rate_limit")
            retry_suggested: Whether retry is suggested
            
        Returns:
            Dictionary with error response
        """
        error_messages = {
            "api_error": "I encountered an issue while processing your request. Please try again in a moment.",
            "timeout": "The request took too long to process. Please try again with a more specific query.",
            "rate_limit": "The service is currently experiencing high demand. Please try again in a few moments.",
            "invalid_request": "I couldn't understand your request. Please rephrase your question.",
            "empty_response": "I couldn't generate a response. Please try rephrasing your question.",
            "unknown": "An unexpected error occurred. Please try again.",
        }
        
        message = error_messages.get(error_type, error_messages["unknown"])
        
        return {
            "text": message,
            "metadata": {
                "is_empty": True,
                "is_complete": False,
                "is_error": True,
                "error_type": error_type,
                "error_message": str(error),
                "retry_suggested": retry_suggested,
            },
        }
