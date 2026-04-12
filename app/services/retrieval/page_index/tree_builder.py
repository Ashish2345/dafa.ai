"""
Tree builder — parses Markdown into a hierarchical tree via LLM.

- Stores character offsets (start_char, end_char) for each node
- Chunked processing for large documents (> _CHUNK_CHARS threshold)
- Progress callbacks are handled by the caller (PageIndexStrategy.ingest)
  so this module stays purely synchronous.
"""

import json
import re
from typing import Any

from loguru import logger

from app.prompts.new_flow import tree_builder as prompts
from app.services.llm import LLMService

# Each chunk sent to the LLM.
# At ~4 chars/token, 350K chars ≈ 87K input tokens.
# Observed output: ~25K tokens per 350K chars chunk → well within 65K limit.
# Gemini 2.5 Flash max output is 65 536 tokens.
_CHUNK_CHARS = 350_000
_MAX_OUTPUT_TOKENS = 65_536


class TreeBuilder:
    """Builds hierarchical document trees from Markdown content."""

    def __init__(self, llm_service: LLMService | None = None):
        self.llm = llm_service or LLMService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, markdown_content: str, language: str = "en", page_bbox_map: list[dict] | None = None) -> dict:
        """
        Build a hierarchical tree from Markdown content (synchronous).

        Large documents are split at heading boundaries into ≤ _CHUNK_CHARS
        chunks; each chunk gets one LLM call; results are merged.

        Progress reporting is intentionally NOT handled here — callers that
        want async progress should use split_chunks() + build_chunk() directly
        (see PageIndexStrategy.ingest).
        """
        if not markdown_content:
            return {"document_title": "Unknown", "language": language, "nodes": []}

        chunks = self.split_chunks(markdown_content)
        if len(chunks) == 1:
            tree = self.build_chunk(chunks[0], language)
        else:
            subtrees = [self.build_chunk(c, language) for c in chunks]
            tree = self._merge_trees(subtrees, language)

        self._attach_char_offsets(tree.get("nodes", []), markdown_content)
        self._assign_node_ids(tree["nodes"])
        if page_bbox_map:
            self._attach_page_bboxes(tree["nodes"], page_bbox_map)
        node_count = self._count_nodes(tree.get("nodes", []))
        logger.info(f"Built tree: {node_count} nodes from {len(markdown_content):,} chars")
        return tree

    def split_chunks(self, content: str) -> list[str]:
        """Split markdown at H1/H2 boundaries into chunks ≤ _CHUNK_CHARS chars."""
        if len(content) <= _CHUNK_CHARS:
            return [content]

        heading_re = re.compile(r"^#{1,2}\s", re.MULTILINE)
        split_positions = [m.start() for m in heading_re.finditer(content)]

        if not split_positions:
            return [content[i : i + _CHUNK_CHARS] for i in range(0, len(content), _CHUNK_CHARS)]

        chunks: list[str] = []
        chunk_start = 0
        for pos in split_positions[1:]:
            if pos - chunk_start >= _CHUNK_CHARS:
                chunks.append(content[chunk_start:pos])
                chunk_start = pos
        chunks.append(content[chunk_start:])
        return chunks

    def build_chunk(self, content: str, language: str) -> dict:
        """Call the LLM for a single chunk and return a parsed tree dict."""
        system_prompt, user_prompt_template = prompts.get_prompts(language)
        user_prompt = user_prompt_template.format(markdown_content=content)
        response = self.llm.call(
            prompt=user_prompt,
            system_instruction=system_prompt,
            temperature=0.1,
            max_tokens=_MAX_OUTPUT_TOKENS,
            add_warning=False,  # never append prose warnings — they corrupt JSON
        )

        tree = self._parse_json_response(response)
        if not tree:
            logger.warning("Tree chunk returned empty/invalid JSON")
            return {"document_title": "Unknown", "language": language, "nodes": []}

        tree["language"] = language
        return tree

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_node_ids(nodes: list[dict], counter: list[int] | None = None) -> None:
        """Assign depth-first traversal IDs to all tree nodes."""
        if counter is None:
            counter = [0]
        for node in nodes:
            node["id"] = counter[0]
            counter[0] += 1
            if node.get("children"):
                TreeBuilder._assign_node_ids(node["children"], counter)

    @staticmethod
    def _attach_page_bboxes(nodes: list[dict], page_bbox_map: list[dict]) -> None:
        """Attach page_bboxes to each node using char offsets or page_range fallback."""
        # Build a lookup: page_number -> bbox entry for fast access
        page_lookup: dict[int, dict] = {}
        for entry in page_bbox_map:
            page_lookup[entry["page"]] = entry

        for node in nodes:
            node_start = node.get("start_char", -1)
            node_end = node.get("end_char", -1)
            page_range = node.get("page_range", [])

            page_bboxes = []

            if node_start >= 0 and node_end >= 0:
                # Primary: use character offsets for proportional bbox slicing
                for entry in page_bbox_map:
                    entry_start = entry["start_char"]
                    entry_end = entry["end_char"]
                    if entry_start < node_end and entry_end > node_start:
                        node_start_in_page = max(node_start, entry_start)
                        node_end_in_page = min(node_end, entry_end)
                        page_char_len = entry_end - entry_start

                        if page_char_len > 0:
                            start_frac = (node_start_in_page - entry_start) / page_char_len
                            end_frac = (node_end_in_page - entry_start) / page_char_len
                        else:
                            start_frac = 0.0
                            end_frac = 1.0

                        bbox = entry["bbox"]
                        page_height = bbox["y2"] - bbox["y0"]

                        page_bboxes.append({
                            "page": entry["page"],
                            "bbox": {
                                "x0": bbox["x0"],
                                "y0": bbox["y0"] + start_frac * page_height,
                                "x2": bbox["x2"],
                                "y2": bbox["y0"] + end_frac * page_height,
                            }
                        })

            elif page_range and len(page_range) >= 1:
                # Fallback: use page_range from LLM when char offsets are missing.
                # Limit to max 2 pages to avoid noisy highlights on broad sections.
                start_page = page_range[0]
                end_page = page_range[-1] if len(page_range) >= 2 else start_page
                end_page = min(end_page, start_page + 1)  # cap at 2 pages
                for pg in range(start_page, end_page + 1):
                    entry = page_lookup.get(pg)
                    if entry:
                        page_bboxes.append({
                            "page": pg,
                            "bbox": entry["bbox"]
                        })

            node["page_bboxes"] = page_bboxes

            if node.get("children"):
                TreeBuilder._attach_page_bboxes(node["children"], page_bbox_map)

    def _merge_trees(self, trees: list[dict], language: str) -> dict:
        title = next(
            (t.get("document_title", "") for t in trees if t.get("document_title", "Unknown") != "Unknown"),
            "Unknown",
        )
        nodes: list[dict] = []
        for t in trees:
            nodes.extend(t.get("nodes", []))
        return {"document_title": title, "language": language, "nodes": nodes}

    def _attach_char_offsets(self, nodes: list, markdown: str) -> None:
        """Record start_char/end_char for each node so text extraction is a simple slice.

        Strategy: collect all node titles, find them in the markdown as plain text,
        then each node's extent runs from its title to the next node's title (or EOF).
        This works for OCR markdown which has no heading markers.
        """
        # Collect all titles across the entire tree (flat) for boundary detection
        all_titles = self._collect_all_titles(nodes)

        # Find positions of all titles in the markdown (for boundary detection)
        title_positions: list[int] = []
        for t in all_titles:
            escaped = re.escape(t)
            # Try heading match first, then plain text
            m = re.search(rf"(#{1,6}[^\n]*{escaped})", markdown, re.IGNORECASE)
            if not m:
                m = re.search(escaped, markdown, re.IGNORECASE)
            if m:
                title_positions.append(m.start())
        title_positions = sorted(set(title_positions))

        self._assign_char_offsets_recursive(nodes, markdown, title_positions)

    @staticmethod
    def _collect_all_titles(nodes: list) -> list[str]:
        """Recursively collect all non-empty titles from the tree."""
        titles = []
        for node in nodes:
            t = node.get("title", "").strip()
            if t:
                titles.append(t)
            titles.extend(TreeBuilder._collect_all_titles(node.get("children", [])))
        return titles

    def _assign_char_offsets_recursive(
        self, nodes: list, markdown: str, title_positions: list[int]
    ) -> None:
        """Assign start_char/end_char for each node using title search + boundary detection."""
        for node in nodes:
            title = node.get("title", "").strip()
            if not title:
                node["start_char"] = -1
                node["end_char"] = -1
                self._assign_char_offsets_recursive(node.get("children", []), markdown, title_positions)
                continue

            escaped = re.escape(title)
            # Try heading match first (## Title), then plain text match
            match = re.search(rf"(#{1,6}[^\n]*{escaped})", markdown, re.IGNORECASE)
            if not match:
                match = re.search(escaped, markdown, re.IGNORECASE)

            if match:
                start = match.start()
                # End = next title's position after this one, or EOF
                end = len(markdown)
                for pos in title_positions:
                    if pos > start + len(title):
                        end = pos
                        break
                node["start_char"] = start
                node["end_char"] = min(end, start + 8000)
            else:
                node["start_char"] = -1
                node["end_char"] = -1

            self._assign_char_offsets_recursive(node.get("children", []), markdown, title_positions)

    def _parse_json_response(self, response: str) -> Any:
        """Strip markdown fences and parse JSON, tolerating trailing garbage."""
        if not response:
            return None
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: extract the outermost { ... } object
        start = text.find("{")
        if start == -1:
            logger.warning(f"No JSON object found in response: {text[:300]}")
            return None

        # Find the matching closing brace by counting depth
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError as exc:
                        logger.warning(f"Failed to parse extracted JSON: {exc}\nResponse: {text[:300]}")
                        return None

        logger.warning(f"Unbalanced braces in response: {text[:300]}")
        return None

    def _count_nodes(self, nodes: list) -> int:
        count = len(nodes)
        for node in nodes:
            count += self._count_nodes(node.get("children", []))
        return count
