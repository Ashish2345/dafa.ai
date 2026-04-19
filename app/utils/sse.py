"""Server-Sent Events helpers — pure formatting, no business logic."""

import json
from typing import Any, Mapping


def format_event(event: str, data: Mapping[str, Any]) -> str:
    """Serialize a single SSE event.

    The trailing blank line is required by the SSE spec to terminate the frame.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def is_rate_limit_error(message: str) -> bool:
    """Heuristic: did this exception text come from a 429/quota failure?"""
    lower = message.lower()
    return (
        "429" in message
        or "RESOURCE_EXHAUSTED" in message
        or "rate limit" in lower
        or "quota" in lower
    )


def rate_limit_event(detail: str = "") -> str:
    """Standard user-facing rate-limit SSE event."""
    return format_event(
        "error",
        {
            "code": "rate_limited",
            "message": (
                "The service is currently experiencing high demand. "
                "Please try again in a few moments."
            ),
            "detail": detail,
        },
    )
