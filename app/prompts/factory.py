"""
Prompt factory — single entry point for all prompts.

Structure:
    prompts/
    ├── factory.py                  ← this file
    └── page_index/                 ← strategy
        ├── en/                     ← English prompts
        │   ├── answer_synthesis.py
        │   ├── tree_navigator.py
        │   └── tree_builder.py
        └── ne/                     ← Nepali prompts
            ├── answer_synthesis.py
            ├── tree_navigator.py
            └── tree_builder.py

Usage:
    from app.prompts.factory import get_prompts

    system, user_template = get_prompts("page_index", "answer_synthesis", "en")
    system, user_template = get_prompts("page_index", "tree_navigator", "ne")
"""

from importlib import import_module
from typing import Optional


def get_prompts(
    strategy: str,
    prompt_name: str,
    language: str = "en",
) -> tuple[str, str]:
    """
    Get (system_prompt, user_prompt_template) by strategy, name, and language.

    Args:
        strategy: "page_index" (or future: "vector_rag")
        prompt_name: "answer_synthesis", "tree_navigator", "tree_builder"
        language: "en" or "ne"

    Returns:
        (SYSTEM, USER) from the resolved module.

    Raises:
        ImportError: If the module doesn't exist.
    """
    lang = language if language in ("en", "ne") else "en"
    module_path = f"app.prompts.{strategy}.{lang}.{prompt_name}"
    mod = import_module(module_path)
    return mod.SYSTEM, mod.USER


def list_prompts(strategy: str = "page_index") -> dict[str, list[str]]:
    """Return available prompt names per language for a strategy."""
    return {
        "en": ["answer_synthesis", "tree_navigator", "tree_builder"],
        "ne": ["answer_synthesis", "tree_navigator", "tree_builder"],
    }
