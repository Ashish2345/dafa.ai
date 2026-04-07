"""
PageIndex Service — Vectorless RAG using hierarchical document tree navigation.

New flow (replaces Qdrant + embeddings + BM25):
  1. build_tree()         — Parse Markdown → LLM generates hierarchical tree with summaries
  2. retrieve_sections()  — LLM navigates tree → returns full text of relevant nodes

Reference: https://github.com/VectifyAI/PageIndex
"""

import json
import re
from typing import Any

from loguru import logger

from app.prompts.new_flow import tree_builder as tree_builder_prompts
from app.prompts.new_flow import tree_navigator as tree_navigator_prompts
from app.services.llm import LLMService


class PageIndexService:
    """
    Builds and queries PageIndex trees for vectorless document retrieval.

    The tree is a hierarchical JSON structure where each node has:
      nodeId, title, summary, page_range, children
    """

    # Maximum Markdown chars sent per LLM call for tree building
    # Gemini 2.5 Flash context window is ~1M tokens; 800K chars ≈ ~200K tokens, safe margin.
    _MAX_MARKDOWN_CHARS = 800_000

    def __init__(self):
        self.llm_service = LLMService()

    # ------------------------------------------------------------------
    # Tree Building
    # ------------------------------------------------------------------

    def build_tree(self, markdown_content: str, language: str = "en") -> dict:
        """
        Build a hierarchical PageIndex tree from Markdown content.

        Args:
            markdown_content: Full Markdown text of the document
            language: "en" or "ne" — controls which prompt variant is used

        Returns:
            Tree dict: {"document_title": str, "language": str, "nodes": [...]}
        """
        if not markdown_content:
            return {"document_title": "Unknown", "language": language, "nodes": []}

        # Truncate if too large (edge case for very long acts)
        content = markdown_content[: self._MAX_MARKDOWN_CHARS]
        if len(markdown_content) > self._MAX_MARKDOWN_CHARS:
            logger.warning(
                f"Markdown truncated from {len(markdown_content)} to {self._MAX_MARKDOWN_CHARS} chars for tree building"
            )

        system_prompt, user_prompt_template = tree_builder_prompts.get_prompts(language)
        user_prompt = user_prompt_template.format(markdown_content=content)

        logger.info(f"Building PageIndex tree (language={language}, chars={len(content)})")

        response = self.llm_service.call(
            prompt=user_prompt,
            system_instruction=system_prompt,
            temperature=0.1,
            max_tokens=8192,
        )

        tree = self._parse_json_response(response)
        if not tree:
            logger.warning("PageIndex tree build returned empty/invalid JSON — returning empty tree")
            return {"document_title": "Unknown", "language": language, "nodes": []}

        # Ensure language is stored in tree
        tree["language"] = language
        node_count = self._count_nodes(tree.get("nodes", []))
        logger.info(f"Built PageIndex tree with {node_count} nodes")
        return tree

    # ------------------------------------------------------------------
    # Section Retrieval
    # ------------------------------------------------------------------

    def retrieve_sections(
        self,
        query: str,
        tree: dict,
        markdown_content: str,
        language: str = "en",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Use LLM to navigate the tree and retrieve relevant section text.

        Args:
            query: User's question
            tree: Tree dict from build_tree()
            markdown_content: Full Markdown (used to extract section text by nodeId)
            language: "en" or "ne"
            top_k: Maximum number of sections to return

        Returns:
            List of dicts: [{nodeId, title, summary, text}, ...]
        """
        if not tree or not tree.get("nodes"):
            logger.warning("Empty tree passed to retrieve_sections")
            return []

        # Build compact tree summary (nodeId + title + summary only, no full text)
        compact_tree = self._build_compact_tree(tree)
        tree_json = json.dumps(compact_tree, ensure_ascii=False, indent=2)

        system_prompt, user_prompt_template = tree_navigator_prompts.get_prompts(language)
        user_prompt = user_prompt_template.format(tree_json=tree_json, query=query)

        logger.info(f"Navigating PageIndex tree for query: {query[:100]!r}")

        response = self.llm_service.call(
            prompt=user_prompt,
            system_instruction=system_prompt,
            temperature=0.1,
            max_tokens=500,
        )

        nav_result = self._parse_json_response(response)
        relevant_node_ids: list[str] = []

        if isinstance(nav_result, dict):
            relevant_node_ids = nav_result.get("relevant_nodes", [])
        elif isinstance(nav_result, list):
            relevant_node_ids = nav_result

        if not relevant_node_ids:
            logger.warning("Tree navigator returned no relevant nodes")
            return []

        # Limit to top_k
        relevant_node_ids = relevant_node_ids[:top_k]
        logger.info(f"Relevant nodeIds: {relevant_node_ids}")

        # Build node lookup from tree
        node_lookup = self._build_node_lookup(tree.get("nodes", []))

        # Extract text for each relevant node from Markdown
        sections = []
        for node_id in relevant_node_ids:
            node = node_lookup.get(node_id)
            if not node:
                logger.debug(f"nodeId {node_id!r} not found in tree")
                continue

            section_text = self._extract_node_text(node_id, node, markdown_content)
            sections.append(
                {
                    "nodeId": node_id,
                    "title": node.get("title", ""),
                    "summary": node.get("summary", ""),
                    "text": section_text,
                    "page_range": node.get("page_range", []),
                    "metadata": {
                        "source": "page_index",
                        "node_id": node_id,
                        "title": node.get("title", ""),
                    },
                }
            )

        return sections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_json_response(self, response: str) -> Any:
        """Strip markdown fences and parse JSON from LLM response."""
        if not response:
            return None
        text = response.strip()
        # Remove ```json ... ``` or ``` ... ```
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse PageIndex JSON response: {exc}\nResponse: {text[:300]}")
            return None

    def _build_compact_tree(self, tree: dict) -> dict:
        """Return tree with only nodeId, title, summary (no page_range, no markdown text)."""

        def _compact_nodes(nodes: list) -> list:
            result = []
            for node in nodes:
                result.append(
                    {
                        "nodeId": node.get("nodeId", ""),
                        "title": node.get("title", ""),
                        "summary": node.get("summary", ""),
                        "children": _compact_nodes(node.get("children", [])),
                    }
                )
            return result

        return {
            "document_title": tree.get("document_title", ""),
            "nodes": _compact_nodes(tree.get("nodes", [])),
        }

    def _build_node_lookup(self, nodes: list, lookup: dict | None = None) -> dict:
        """Recursively build a flat nodeId → node dict for fast lookup."""
        if lookup is None:
            lookup = {}
        for node in nodes:
            node_id = node.get("nodeId", "")
            if node_id:
                lookup[node_id] = node
            self._build_node_lookup(node.get("children", []), lookup)
        return lookup

    def _count_nodes(self, nodes: list) -> int:
        count = len(nodes)
        for node in nodes:
            count += self._count_nodes(node.get("children", []))
        return count

    def _extract_node_text(self, node_id: str, node: dict, markdown: str) -> str:
        """
        Extract the text corresponding to a node from the full Markdown.

        Strategy: search for the section heading (node title) in Markdown and
        extract until the next same-level heading. Falls back to node summary.
        """
        title = node.get("title", "").strip()
        if not title or not markdown:
            return node.get("summary", "")

        # Escape title for regex
        escaped = re.escape(title)

        # Try to find the section in markdown (heading or plain text)
        pattern = rf"(#{1,4}\s*{escaped}.*?)(?=\n#{1,4}\s|\Z)"
        match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
            # Limit to 3000 chars to avoid huge context
            return text[:3000]

        # Fallback: search for title as a plain line
        lines = markdown.split("\n")
        for i, line in enumerate(lines):
            if title.lower() in line.lower():
                # Extract next 50 lines
                snippet = "\n".join(lines[i : i + 50])
                return snippet[:3000]

        return node.get("summary", "")
