"""
Section retriever — LLM-guided tree navigation + text extraction.

Fixes over original:
- Uses character offsets instead of fragile regex
- Deduplicates nodeIds before text extraction
"""

import json
import re
import time
from typing import Any

from loguru import logger

from app.prompts.factory import get_prompts
from app.services.llm import LLMService


class SectionRetriever:
    """Navigates a PageIndex tree to find and extract relevant sections."""

    def __init__(self, llm_service: LLMService | None = None):
        self.llm = llm_service or LLMService()

    def retrieve(
        self,
        query: str,
        tree: dict,
        markdown_content: str,
        language: str = "en",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Navigate tree and retrieve relevant section text.

        Returns:
            [{nodeId, title, summary, text, page_range, metadata}, ...]
        """
        if not tree or not tree.get("nodes"):
            return []

        compact_tree = self._build_compact_tree(tree)
        tree_json = json.dumps(compact_tree, ensure_ascii=False, indent=2)

        system_prompt, user_prompt_template = get_prompts("page_index", "tree_navigator", language)
        user_prompt = user_prompt_template.format(tree_json=tree_json, query=query)

        logger.info(f"Navigating tree for query: {query[:100]!r}")

        nav_start = time.time()
        response = self.llm.call(
            prompt=user_prompt,
            system_instruction=system_prompt,
            temperature=0.1,
            max_tokens=4096,  # thinking tokens eat into the budget; 500 is too low
            response_mime_type="application/json",
            add_warning=False,
        )
        logger.info(f"[latency] navigator LLM call: {time.time() - nav_start:.2f}s")

        nav_result = self._parse_json_response(response)
        relevant_node_ids: list[str] = []

        if isinstance(nav_result, dict):
            relevant_node_ids = nav_result.get("relevant_nodes", [])
        elif isinstance(nav_result, list):
            relevant_node_ids = nav_result

        if not relevant_node_ids:
            logger.warning("Tree navigator returned no relevant nodes")
            return []

        # Deduplicate preserving order
        seen = set()
        unique_ids = []
        for nid in relevant_node_ids:
            if nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)
        relevant_node_ids = unique_ids[:top_k]

        logger.info(f"Relevant nodeIds: {relevant_node_ids}")

        node_lookup = self._build_node_lookup(tree.get("nodes", []))

        sections = []
        for node_id in relevant_node_ids:
            node = node_lookup.get(node_id)
            if not node:
                logger.debug(f"nodeId {node_id!r} not found in tree")
                continue

            section_text, is_exact = self._extract_node_text(node, markdown_content)
            page_range = node.get("page_range", [])
            sections.append(
                {
                    "nodeId": node_id,
                    "int_id": node.get("id"),
                    "title": node.get("title", ""),
                    "summary": node.get("summary", ""),
                    "text": section_text,
                    "is_exact_text": is_exact,
                    "page_range": page_range,
                    "page_bboxes": node.get("page_bboxes", []),
                    "metadata": {
                        "source": "page_index",
                        "node_id": node_id,
                        "title": node.get("title", ""),
                    },
                }
            )

        return sections

    def _extract_node_text(self, node: dict, markdown: str) -> tuple[str, bool]:
        """
        Extract text using char offsets. Falls back to regex then summary.

        Returns:
            (text, is_exact) — is_exact=True when text is from the actual document,
            False when it's an LLM-generated summary (fallback).
        """
        start = node.get("start_char", -1)
        end = node.get("end_char", -1)

        if start >= 0 and end > start and start < len(markdown):
            text = markdown[start : min(end, len(markdown))]
            # Strip line-number markers (->N<-) left by OCR processing
            # Match ``-> N<-`` including padding whitespace inside the delimiters
            # (OCR parser right-justifies the number, e.g. ``-> 6<-``).
            text = re.sub(r"->\s*\d+\s*<-\s*", "", text)
            return text, True

        # Fallback: regex (for trees built before offset support)
        title = node.get("title", "").strip()
        if title and markdown:
            escaped = re.escape(title)
            pattern = rf"(#{1,6}[^\n]*{escaped}.*?)(?=\n#{1,6}\s|\Z)"
            match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()[:3000], True

        # Last resort — LLM summary, not exact document text
        return node.get("summary", ""), False

    def _build_compact_tree(self, tree: dict) -> dict:
        def _compact(nodes: list) -> list:
            return [
                {
                    "nodeId": n.get("nodeId", ""),
                    "title": n.get("title", ""),
                    "summary": n.get("summary", ""),
                    "children": _compact(n.get("children", [])),
                }
                for n in nodes
            ]

        return {
            "document_title": tree.get("document_title", ""),
            "nodes": _compact(tree.get("nodes", [])),
        }

    def _build_node_lookup(self, nodes: list, lookup: dict | None = None) -> dict:
        if lookup is None:
            lookup = {}
        for node in nodes:
            nid = node.get("nodeId", "")
            if nid:
                lookup[nid] = node
            self._build_node_lookup(node.get("children", []), lookup)
        return lookup

    def _parse_json_response(self, response: str) -> Any:
        if not response:
            return None
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse navigator JSON: {exc}")
            return None
