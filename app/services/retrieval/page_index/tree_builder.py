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
            # Force native JSON mode: Gemini stops exploring formatting and
            # spends far fewer thinking tokens. Critical on Gemini 2.5 Flash
            # which otherwise burns 40K+ thinking tokens on Nepali legal JSON.
            response_mime_type="application/json",
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

    def _attach_char_offsets(
        self, nodes: list, markdown: str, page_bbox_map: list[dict] | None = None,
    ) -> list[dict]:
        """Record start_char/end_char for each node so text extraction is a simple slice.

        Two-pass approach:
          Pass 1: Assign start_char to every node by searching for its title in
                  the markdown, constrained to the node's page_range region.
          Pass 2: Collect all assigned start_chars as boundary positions, then
                  set each node's end_char to the next boundary (or EOF).

        Returns:
            A list of ``{"nodeId": ..., "title": ...}`` dicts for every node
            whose title was NOT found in the markdown (``start_char == -1``).
            The LLM-generated path ignores the return value; the custom-tree
            path uses it to populate ``ingest_warnings``.
        """
        # Build page → char range lookup for constraining searches
        page_char_ranges: dict[int, tuple[int, int]] = {}
        if page_bbox_map:
            for entry in page_bbox_map:
                page_char_ranges[entry["page"]] = (entry["start_char"], entry["end_char"])

        # Pass 1: assign start_char to each node
        self._assign_start_chars(nodes, markdown, page_char_ranges)

        # Pass 2: collect all start positions, then assign end_char
        all_starts = sorted(set(self._collect_start_chars(nodes)))
        self._assign_end_chars(nodes, markdown, all_starts)

        # Pass 3: collect nodeIds whose titles were not found
        return self._collect_unmatched(nodes)

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

    def _get_page_region(
        self, node: dict, page_char_ranges: dict[int, tuple[int, int]], md_len: int,
    ) -> tuple[int, int]:
        """Return (region_start, region_end) in the markdown for this node's page_range."""
        page_range = node.get("page_range", [])
        if not page_char_ranges or not page_range:
            return 0, md_len
        first_page = page_range[0]
        last_page = page_range[-1] if len(page_range) >= 2 else first_page
        # Clamp to valid page range
        max_page = max(page_char_ranges.keys()) if page_char_ranges else 0
        first_page = max(1, first_page)
        last_page = min(last_page, max_page)
        if first_page > last_page:
            return 0, md_len
        starts, ends = [], []
        for pg in range(first_page, last_page + 1):
            if pg in page_char_ranges:
                s, e = page_char_ranges[pg]
                starts.append(s)
                ends.append(e)
        if starts:
            return min(starts), max(ends)
        return 0, md_len

    def _assign_start_chars(
        self, nodes: list, markdown: str, page_char_ranges: dict[int, tuple[int, int]],
    ) -> None:
        """Pass 1: find each node's title in its page region and set start_char."""
        for node in nodes:
            title = node.get("title", "").strip()
            if not title:
                node["start_char"] = -1
                node["end_char"] = -1
                self._assign_start_chars(node.get("children", []), markdown, page_char_ranges)
                continue

            escaped = re.escape(title)
            region_start, region_end = self._get_page_region(node, page_char_ranges, len(markdown))
            region_text = markdown[region_start:region_end]

            match = re.search(rf"(#{1,6}[^\n]*{escaped})", region_text, re.IGNORECASE)
            if not match:
                match = re.search(escaped, region_text, re.IGNORECASE)

            if match:
                node["start_char"] = region_start + match.start()
            else:
                node["start_char"] = -1

            node["end_char"] = -1  # filled in pass 2
            self._assign_start_chars(node.get("children", []), markdown, page_char_ranges)

    def _collect_start_chars(self, nodes: list) -> list[int]:
        """Collect all assigned start_char values from the tree (for boundary detection)."""
        result = []
        for node in nodes:
            sc = node.get("start_char", -1)
            if sc >= 0:
                result.append(sc)
            result.extend(self._collect_start_chars(node.get("children", [])))
        return result

    def _collect_unmatched(self, nodes: list) -> list[dict]:
        """Recursively collect ``{nodeId, title}`` for nodes with start_char == -1."""
        result: list[dict] = []
        for node in nodes:
            if node.get("start_char", -1) < 0:
                result.append({
                    "nodeId": node.get("nodeId", ""),
                    "title": node.get("title", ""),
                })
            result.extend(self._collect_unmatched(node.get("children", [])))
        return result

    def _assign_end_chars(self, nodes: list, markdown: str, all_starts: list[int]) -> None:
        """Pass 2: set end_char = next node's start_char (from all_starts), capped at 8000."""
        for node in nodes:
            start = node.get("start_char", -1)
            if start >= 0:
                # Find the next boundary after this node's title
                end = len(markdown)
                title_len = len(node.get("title", ""))
                for pos in all_starts:
                    if pos > start + title_len:
                        end = pos
                        break
                node["end_char"] = end
            else:
                node["end_char"] = -1

            self._assign_end_chars(node.get("children", []), markdown, all_starts)

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
