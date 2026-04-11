"""
Tree builder — parses Markdown into a hierarchical tree via LLM.

Migrated from app/services/page_index/service.py with fixes:
- Stores character offsets (start_char, end_char) for each node
"""

import json
import re
from typing import Any

from loguru import logger

from app.prompts.new_flow import tree_builder as prompts
from app.services.llm import LLMService


class TreeBuilder:
    """Builds hierarchical document trees from Markdown content."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        max_markdown_chars: int = 800_000,
    ):
        self.llm = llm_service or LLMService()
        self.max_markdown_chars = max_markdown_chars

    def build(self, markdown_content: str, language: str = "en") -> dict:
        """
        Build a hierarchical tree from Markdown content.

        Returns:
            Tree dict: {"document_title": str, "language": str, "nodes": [...]}
        """
        if not markdown_content:
            return {"document_title": "Unknown", "language": language, "nodes": []}

        content = markdown_content[: self.max_markdown_chars]
        if len(markdown_content) > self.max_markdown_chars:
            logger.warning(
                f"Markdown truncated from {len(markdown_content)} to {self.max_markdown_chars} chars"
            )

        system_prompt, user_prompt_template = prompts.get_prompts(language)
        user_prompt = user_prompt_template.format(markdown_content=content)

        logger.info(f"Building PageIndex tree (language={language}, chars={len(content)})")

        response = self.llm.call(
            prompt=user_prompt,
            system_instruction=system_prompt,
            temperature=0.1,
            max_tokens=8192,
        )

        tree = self._parse_json_response(response)
        if not tree:
            logger.warning("Tree build returned empty/invalid JSON")
            return {"document_title": "Unknown", "language": language, "nodes": []}

        tree["language"] = language
        self._attach_char_offsets(tree.get("nodes", []), content)

        node_count = self._count_nodes(tree.get("nodes", []))
        logger.info(f"Built tree with {node_count} nodes")
        return tree

    def _attach_char_offsets(self, nodes: list, markdown: str) -> None:
        """Record start_char/end_char for each node so text extraction is a simple slice."""
        for node in nodes:
            title = node.get("title", "").strip()
            if not title:
                node["start_char"] = -1
                node["end_char"] = -1
                self._attach_char_offsets(node.get("children", []), markdown)
                continue

            escaped = re.escape(title)
            pattern = rf"(#{1,6}\s*{escaped})"
            match = re.search(pattern, markdown, re.IGNORECASE)

            if match:
                start = match.start()
                heading_match = re.match(r"^(#{1,6})\s", match.group(0))
                if heading_match:
                    level = len(heading_match.group(1))
                    next_heading = re.compile(rf"^#{{{1},{level}}}\s", re.MULTILINE)
                    end_match = next_heading.search(markdown, match.end())
                    end = end_match.start() if end_match else len(markdown)
                else:
                    end = len(markdown)
                node["start_char"] = start
                node["end_char"] = min(end, start + 5000)
            else:
                node["start_char"] = -1
                node["end_char"] = -1

            self._attach_char_offsets(node.get("children", []), markdown)

    def _parse_json_response(self, response: str) -> Any:
        """Strip markdown fences and parse JSON."""
        if not response:
            return None
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse tree JSON: {exc}\nResponse: {text[:300]}")
            return None

    def _count_nodes(self, nodes: list) -> int:
        count = len(nodes)
        for node in nodes:
            count += self._count_nodes(node.get("children", []))
        return count
