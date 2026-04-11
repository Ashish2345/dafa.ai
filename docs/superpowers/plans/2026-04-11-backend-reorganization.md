# Backend Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the backend around a strategy pattern for dual RAG approaches (PageIndex + Vector), consolidate API routes from 8 to 4, remove dead code, and fix PageIndex bugs.

**Architecture:** Common `RetrievalStrategy` ABC with `retrieve()` and `ingest()` methods. Two implementations: `PageIndexStrategy` (vectorless, default) and `VectorStrategy` (embeddings + Qdrant). A factory selects the strategy per-request. Thin `QueryOrchestrator` wires strategy to LLM. Consolidated `IngestionPipeline` delegates storage to the strategy.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, MongoDB (motor/pymongo), Qdrant, OpenAI embeddings, Google Gemini 2.5 Flash, LlamaIndex (chunking), sentence-transformers (re-ranking)

**Spec:** `docs/superpowers/specs/2026-04-11-backend-reorganization-design.md`

---

## File Map

### New files to create
| File | Responsibility |
|------|---------------|
| `app/services/retrieval/base.py` | `RetrievalStrategy` ABC + `RetrievedChunk` dataclass |
| `app/services/retrieval/factory.py` | `RetrievalFactory` — picks strategy from name/settings |
| `app/services/retrieval/page_index/__init__.py` | Package init |
| `app/services/retrieval/page_index/strategy.py` | `PageIndexStrategy` — implements retrieve + ingest |
| `app/services/retrieval/page_index/tree_builder.py` | Tree building from markdown via LLM |
| `app/services/retrieval/page_index/section_retriever.py` | Tree navigation + text extraction |
| `app/services/retrieval/vector/__init__.py` | Package init |
| `app/services/retrieval/vector/strategy.py` | `VectorStrategy` — implements retrieve + ingest |
| `app/services/ingestion/pipeline.py` | Unified ingestion pipeline |
| `app/services/ingestion/processing.py` | Re-exports DocumentProcessor + MetadataExtractor |
| `app/services/ingestion/chunking.py` | Copy of processing/chunking.py |
| `app/services/query/__init__.py` | Package init |
| `app/services/query/orchestrator.py` | Thin query orchestrator |
| `app/db/repositories/document_repository.py` | Consolidated document CRUD |
| `app/api/v1/endpoints/query.py` | Query endpoint (replaces ask.py) |

### Existing files to modify
| File | Changes |
|------|---------|
| `app/models/enums.py` | Add `RetrievalStrategyType` enum |
| `app/settings.py` | Replace `use_page_index` with `default_retrieval_strategy`, remove dead flags |
| `app/db/mongodb.py:89` | Remove hardcoded connection string |
| `app/api/v1/endpoints/documents.py` | Rewrite: merge ingest, collections, files into document CRUD |
| `app/api/v1/router.py` | Update to 4 route groups |
| `app/services/llm/service.py` | Add `synthesize()` method |
| `app/services/retrieval/__init__.py` | Update exports |
| `app/services/ingestion/__init__.py` | Update exports |

### Files to delete (Task 13)
| File | Reason |
|------|--------|
| `app/services/rag/` (entire dir) | Replaced by `query/` + `retrieval/` |
| `app/services/page_index/` (entire dir) | Migrated to `retrieval/page_index/` |
| `app/services/embeddings/` (entire dir) | Moved to `retrieval/vector/` |
| `app/services/vector_store/` (entire dir) | Moved to `retrieval/vector/` |
| `app/services/processing/` (entire dir) | Absorbed into `ingestion/` |
| `app/services/download/` (entire dir) | Inlined into route |
| `app/services/retrieval/query_enhancer.py` | Dead code |
| `app/services/retrieval/metadata_enhancer.py` | Dead feature |
| `app/services/retrieval/context_optimizer.py` | Inlined |
| `app/services/retrieval/page_index_retriever.py` | Migrated |
| `app/services/retrieval/hybrid_search.py` | Moved to `vector/` |
| `app/services/retrieval/bm25_search.py` | Moved to `vector/` |
| `app/services/retrieval/reranker.py` | Moved to `vector/` |
| `app/services/ingestion/service.py` | Replaced by `pipeline.py` |
| `app/api/v1/endpoints/ask.py` | Replaced by `query.py` |
| `app/api/v1/endpoints/parse.py` | Redundant |
| `app/api/v1/endpoints/collections.py` | Merged into `documents.py` |
| `app/api/v1/endpoints/files.py` | Merged into `documents.py` |
| `app/api/v1/endpoints/ingest.py` | Merged into `documents.py` |
| `app/prompts/old_flow/` | Legacy prompts |

---

## Task 1: Add RetrievalStrategyType enum and update settings

**Files:**
- Modify: `app/models/enums.py`
- Modify: `app/settings.py`

- [ ] **Step 1: Add the enum to `app/models/enums.py`**

Add after `OCRProvider` class:

```python
class RetrievalStrategyType(str, Enum):
    """Retrieval strategy for RAG queries."""

    PAGE_INDEX = "page_index"
    VECTOR = "vector"
```

- [ ] **Step 2: Update `app/settings.py`**

Update import line:
```python
from app.models.enums import Environment, StorageBackend, RetrievalStrategyType
```

Replace `use_page_index` field with:
```python
    default_retrieval_strategy: RetrievalStrategyType = Field(
        default=RetrievalStrategyType.PAGE_INDEX,
        description="Default retrieval strategy: 'page_index' (vectorless RAG) or 'vector' (Qdrant+embeddings). "
        "Can be overridden per-request.",
    )
```

Remove `rerank_use_llm` field (dead — LLMReranker never enabled in practice).

- [ ] **Step 3: Verify import**

Run: `python -c "from app.settings import settings; print(settings.default_retrieval_strategy)"`
Expected: `page_index`

- [ ] **Step 4: Commit**

```bash
git add app/models/enums.py app/settings.py
git commit -m "feat: add RetrievalStrategyType enum, replace use_page_index setting"
```

---

## Task 2: Fix hardcoded MongoDB connection string

**Files:**
- Modify: `app/db/mongodb.py`

- [ ] **Step 1: Delete line 89 in `app/db/mongodb.py`**

Remove this line:
```python
            mongodb_url = "mongodb://test:test@localhost:27017/?authSource=dafai"
```

This overrides the entire connection URL construction logic above it.

- [ ] **Step 2: Verify import**

Run: `python -c "from app.db.mongodb import MongoDB; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add app/db/mongodb.py
git commit -m "fix: remove hardcoded MongoDB connection string, use settings"
```

---

## Task 3: Create RetrievalStrategy ABC and RetrievedChunk

**Files:**
- Create: `app/services/retrieval/base.py`

- [ ] **Step 1: Create `app/services/retrieval/base.py`**

```python
"""
Retrieval strategy interface.

All retrieval approaches (PageIndex, Vector) implement this ABC.
The factory selects the right implementation per-request.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """Common output shape for all retrieval strategies."""

    text: str
    source: dict[str, Any]
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalStrategy(ABC):
    """
    Abstract base for retrieval strategies.

    Each strategy implements both retrieval and ingestion because
    storage format is strategy-specific (trees vs embeddings).
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return ranked chunks relevant to the query."""
        ...

    @abstractmethod
    async def ingest(
        self,
        document_id: str,
        markdown: str,
        metadata: dict[str, Any],
    ) -> None:
        """Store a processed document for later retrieval."""
        ...
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.services.retrieval.base import RetrievalStrategy, RetrievedChunk; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add app/services/retrieval/base.py
git commit -m "feat: add RetrievalStrategy ABC and RetrievedChunk dataclass"
```

---

## Task 4: Create PageIndex tree_builder and section_retriever

**Files:**
- Create: `app/services/retrieval/page_index/__init__.py`
- Create: `app/services/retrieval/page_index/tree_builder.py`
- Create: `app/services/retrieval/page_index/section_retriever.py`

- [ ] **Step 1: Create package init**

Create `app/services/retrieval/page_index/__init__.py`:
```python
"""PageIndex (vectorless RAG) retrieval strategy."""
```

- [ ] **Step 2: Create `tree_builder.py`**

Migrated from `app/services/page_index/service.py` with char-offset fix.

```python
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
```

- [ ] **Step 3: Create `section_retriever.py`**

Migrated from `app/services/page_index/service.py` with deduplication fix and char-offset extraction.

```python
"""
Section retriever — LLM-guided tree navigation + text extraction.

Fixes over original:
- Uses character offsets instead of fragile regex
- Deduplicates nodeIds before text extraction
"""

import json
import re
from typing import Any

from loguru import logger

from app.prompts.new_flow import tree_navigator as prompts
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

        system_prompt, user_prompt_template = prompts.get_prompts(language)
        user_prompt = user_prompt_template.format(tree_json=tree_json, query=query)

        logger.info(f"Navigating tree for query: {query[:100]!r}")

        response = self.llm.call(
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

            section_text = self._extract_node_text(node, markdown_content)
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

    def _extract_node_text(self, node: dict, markdown: str) -> str:
        """Extract text using char offsets. Falls back to regex then summary."""
        start = node.get("start_char", -1)
        end = node.get("end_char", -1)

        if start >= 0 and end > start and start < len(markdown):
            return markdown[start : min(end, len(markdown))]

        # Fallback: regex (for trees built before offset support)
        title = node.get("title", "").strip()
        if title and markdown:
            escaped = re.escape(title)
            pattern = rf"(#{1,6}\s*{escaped}.*?)(?=\n#{1,6}\s|\Z)"
            match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()[:3000]

        return node.get("summary", "")

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
```

- [ ] **Step 4: Verify imports**

Run: `python -c "from app.services.retrieval.page_index.tree_builder import TreeBuilder; from app.services.retrieval.page_index.section_retriever import SectionRetriever; print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add app/services/retrieval/page_index/
git commit -m "feat: add PageIndex tree_builder and section_retriever with char-offset extraction"
```

---

## Task 5: Create PageIndexStrategy

**Files:**
- Create: `app/services/retrieval/page_index/strategy.py`

- [ ] **Step 1: Create `strategy.py`**

```python
"""
PageIndex retrieval strategy — vectorless RAG.

Implements RetrievalStrategy using LLM-guided hierarchical tree navigation.
"""

from typing import Any

from loguru import logger

from app.db.repositories.page_index_repository import PageIndexRepository
from app.services.llm import LLMService
from app.services.retrieval.base import RetrievalStrategy, RetrievedChunk
from app.services.retrieval.page_index.section_retriever import SectionRetriever
from app.services.retrieval.page_index.tree_builder import TreeBuilder


class PageIndexStrategy(RetrievalStrategy):
    """Vectorless RAG using hierarchical document tree navigation."""

    def __init__(
        self,
        repository: PageIndexRepository,
        tree_builder: TreeBuilder | None = None,
        section_retriever: SectionRetriever | None = None,
        llm_service: LLMService | None = None,
    ):
        self.repo = repository
        llm = llm_service or LLMService()
        self.tree_builder = tree_builder or TreeBuilder(llm_service=llm)
        self.section_retriever = section_retriever or SectionRetriever(llm_service=llm)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[RetrievedChunk]:
        document_ids = await self.repo.list_document_ids()
        if not document_ids:
            logger.warning("No PageIndex trees found")
            return []

        if collection_name:
            document_ids = await self._filter_by_collection(document_ids, collection_name)

        all_chunks: list[RetrievedChunk] = []

        for doc_id in document_ids:
            tree_doc = await self.repo.get_tree(doc_id)
            markdown = await self.repo.get_markdown(doc_id)
            if not tree_doc or not markdown:
                continue

            tree = tree_doc.get("tree", {})
            language = tree_doc.get("language", "en")
            act_name = tree.get("document_title", doc_id)

            sections = self.section_retriever.retrieve(
                query=query,
                tree=tree,
                markdown_content=markdown,
                language=language,
                top_k=top_k,
            )

            for idx, section in enumerate(sections):
                all_chunks.append(
                    RetrievedChunk(
                        text=section["text"],
                        source={
                            "document_name": act_name,
                            "document_id": doc_id,
                            "page_range": section.get("page_range", []),
                            "section": section.get("title", ""),
                            "node_id": section["nodeId"],
                        },
                        score=1.0 - (idx * 0.1),
                        metadata={
                            "act_name": act_name,
                            "document_id": doc_id,
                            "node_id": section["nodeId"],
                            "source": "page_index",
                        },
                    )
                )

        all_chunks.sort(key=lambda c: c.score, reverse=True)
        return all_chunks[:top_k]

    async def ingest(
        self,
        document_id: str,
        markdown: str,
        metadata: dict[str, Any],
    ) -> None:
        language = metadata.get("language", "en")
        logger.info(f"PageIndex ingesting document_id={document_id}")

        tree = self.tree_builder.build(markdown, language=language)
        await self.repo.save_tree(
            document_id=document_id,
            tree=tree,
            markdown=markdown,
            language=language,
        )
        logger.info(f"PageIndex tree saved: {self.tree_builder._count_nodes(tree.get('nodes', []))} nodes")

    async def _filter_by_collection(self, document_ids: list[str], collection_name: str) -> list[str]:
        matched = []
        normalized = collection_name.lower().replace("_", " ")
        for doc_id in document_ids:
            tree_doc = await self.repo.get_tree(doc_id)
            if not tree_doc:
                continue
            title = tree_doc.get("tree", {}).get("document_title", "")
            if normalized in title.lower():
                matched.append(doc_id)
        return matched if matched else document_ids
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.services.retrieval.page_index.strategy import PageIndexStrategy; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add app/services/retrieval/page_index/strategy.py
git commit -m "feat: add PageIndexStrategy implementing RetrievalStrategy"
```

---

## Task 6: Move vector RAG code into retrieval/vector/

**Files:**
- Create: `app/services/retrieval/vector/__init__.py`
- Copy: `services/embeddings/service.py` -> `retrieval/vector/embeddings.py`
- Copy: `services/vector_store/service.py` -> `retrieval/vector/vector_store.py`
- Copy: `services/retrieval/hybrid_search.py` -> `retrieval/vector/hybrid_search.py`
- Copy: `services/retrieval/bm25_search.py` -> `retrieval/vector/bm25_search.py`
- Copy: `services/retrieval/reranker.py` -> `retrieval/vector/reranker.py`

- [ ] **Step 1: Create vector package and copy files**

```bash
cd /home/aashish/Desktop/dafa.ai/backend
mkdir -p app/services/retrieval/vector
```

Create `app/services/retrieval/vector/__init__.py`:
```python
"""Vector-based RAG retrieval strategy (Qdrant + embeddings + BM25 hybrid)."""
```

```bash
cp app/services/embeddings/service.py app/services/retrieval/vector/embeddings.py
cp app/services/vector_store/service.py app/services/retrieval/vector/vector_store.py
cp app/services/retrieval/hybrid_search.py app/services/retrieval/vector/hybrid_search.py
cp app/services/retrieval/bm25_search.py app/services/retrieval/vector/bm25_search.py
cp app/services/retrieval/reranker.py app/services/retrieval/vector/reranker.py
```

- [ ] **Step 2: Update imports in copied files**

In `app/services/retrieval/vector/hybrid_search.py`:
- `from app.services.vector_store import VectorStoreService` -> `from app.services.retrieval.vector.vector_store import VectorStoreService`
- `from app.services.retrieval.bm25_search import BM25SearchService` -> `from app.services.retrieval.vector.bm25_search import BM25SearchService`

In `app/services/retrieval/vector/reranker.py`:
- Keep `from app.services.llm import LLMService` as-is (LLM stays in place)

Check all other cross-references and update to new paths.

- [ ] **Step 3: Verify imports**

Run: `python -c "from app.services.retrieval.vector.embeddings import EmbeddingService; from app.services.retrieval.vector.vector_store import VectorStoreService; from app.services.retrieval.vector.hybrid_search import HybridSearchService; from app.services.retrieval.vector.bm25_search import BM25SearchService; from app.services.retrieval.vector.reranker import Reranker; print('All vector imports OK')"`

- [ ] **Step 4: Commit**

```bash
git add app/services/retrieval/vector/
git commit -m "feat: move vector RAG code into retrieval/vector/ package"
```

---

## Task 7: Create VectorStrategy

**Files:**
- Create: `app/services/retrieval/vector/strategy.py`

- [ ] **Step 1: Create `strategy.py`**

```python
"""
Vector retrieval strategy — traditional embeddings + Qdrant + BM25 hybrid.
"""

import re
from typing import Any

from loguru import logger

from app.services.retrieval.base import RetrievalStrategy, RetrievedChunk
from app.services.retrieval.vector.bm25_search import BM25SearchService
from app.services.retrieval.vector.embeddings import EmbeddingService
from app.services.retrieval.vector.hybrid_search import HybridSearchService
from app.services.retrieval.vector.reranker import Reranker
from app.services.retrieval.vector.vector_store import VectorStoreService


class VectorStrategy(RetrievalStrategy):
    """Traditional embeddings + vector search + BM25 hybrid + re-ranking."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStoreService | None = None,
        hybrid_search: HybridSearchService | None = None,
        reranker: Reranker | None = None,
        chunking_service: Any | None = None,
    ):
        self.embeddings = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreService()

        if hybrid_search:
            self.hybrid_search = hybrid_search
        else:
            from app.settings import settings
            bm25 = BM25SearchService()
            self.hybrid_search = HybridSearchService(
                vector_store=self.vector_store,
                bm25_service=bm25,
                vector_weight=getattr(settings, "hybrid_search_vector_weight", 0.7),
                bm25_weight=getattr(settings, "hybrid_search_bm25_weight", 0.3),
            )

        if reranker is not None:
            self.reranker = reranker
        else:
            from app.settings import settings
            self.reranker = Reranker(
                model_name=getattr(settings, "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                use_cross_encoder=True,
                top_k=getattr(settings, "rerank_top_k", 5),
            )

        self.chunking_service = chunking_service

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[RetrievedChunk]:
        query_embeddings = self.embeddings.generate_embeddings([query])
        if not query_embeddings or not query_embeddings[0]:
            logger.error("Failed to generate query embedding")
            return []
        query_embedding = query_embeddings[0]

        if collection_name:
            collection_names = [collection_name]
        else:
            collection_names = self.vector_store.get_all_collections()
            if not collection_names:
                return []

        from app.settings import settings
        retrieve_k = getattr(settings, "rerank_retrieve_k", 20)

        for cname in collection_names:
            if not self.hybrid_search.bm25_service.indexes.get(cname):
                self.hybrid_search.load_bm25_index_from_qdrant(cname)

        search_results = self.hybrid_search.search(
            query=query,
            query_embedding=query_embedding,
            collection_names=collection_names,
            limit=retrieve_k,
            filter_conditions=filter_conditions,
            retrieve_k=retrieve_k * 2,
        )

        if not search_results:
            return []

        if self.reranker:
            search_results = self.reranker.rerank(query=query, chunks=search_results, top_k=top_k)
        else:
            search_results = search_results[:top_k]

        return [
            RetrievedChunk(
                text=r.get("text", ""),
                source={
                    "document_name": r.get("metadata", {}).get("act_name", "Unknown"),
                    "section": ", ".join(r.get("metadata", {}).get("sections_in_chunk", [])),
                    "chunk_id": r.get("chunk_id", ""),
                },
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            )
            for r in search_results
        ]

    async def ingest(
        self,
        document_id: str,
        markdown: str,
        metadata: dict[str, Any],
    ) -> None:
        from app.services.ingestion.chunking import ChunkingService

        chunking = self.chunking_service or ChunkingService()
        chunks = chunking.chunk_document(markdown, metadata)

        chunk_texts = [c.get("text", "") for c in chunks]
        embeddings = self.embeddings.generate_embeddings(chunk_texts)
        if not embeddings:
            logger.error("Failed to generate embeddings during ingestion")
            return

        act_name = metadata.get("act_name", "")
        collection_name = re.sub(r"[^a-zA-Z0-9\s]", "", act_name).lower().strip()
        collection_name = re.sub(r"\s+", "_", collection_name) or "unknown_act"

        self.vector_store.store_chunks(
            chunks=chunks, embeddings=embeddings,
            document_id=document_id, collection_name=collection_name,
        )
        self.hybrid_search.build_bm25_index(chunks=chunks, collection_name=collection_name)
        logger.info(f"Vector ingestion complete: {len(chunks)} chunks in '{collection_name}'")
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.services.retrieval.vector.strategy import VectorStrategy; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add app/services/retrieval/vector/strategy.py
git commit -m "feat: add VectorStrategy implementing RetrievalStrategy"
```

---

## Task 8: Create RetrievalFactory

**Files:**
- Create: `app/services/retrieval/factory.py`

- [ ] **Step 1: Create `factory.py`**

```python
"""
Retrieval strategy factory.

Selects the right strategy based on request param or system default.
"""

from app.db.mongodb import get_database
from app.db.repositories.page_index_repository import PageIndexRepository
from app.services.retrieval.base import RetrievalStrategy
from app.settings import settings


class RetrievalFactory:
    @staticmethod
    async def get_strategy(strategy_name: str | None = None) -> RetrievalStrategy:
        """
        Return the appropriate RetrievalStrategy.

        Args:
            strategy_name: "page_index" or "vector". Defaults to settings.default_retrieval_strategy.
        """
        name = strategy_name or settings.default_retrieval_strategy

        if name == "page_index":
            from app.services.retrieval.page_index.strategy import PageIndexStrategy
            db = await get_database()
            repo = PageIndexRepository(db)
            return PageIndexStrategy(repository=repo)

        elif name == "vector":
            from app.services.retrieval.vector.strategy import VectorStrategy
            return VectorStrategy()

        raise ValueError(f"Unknown strategy: {name!r}. Use 'page_index' or 'vector'.")
```

- [ ] **Step 2: Verify import**

Run: `python -c "from app.services.retrieval.factory import RetrievalFactory; print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add app/services/retrieval/factory.py
git commit -m "feat: add RetrievalFactory for strategy selection"
```

---

## Task 9: Add synthesize() to LLMService and create QueryOrchestrator

**Files:**
- Modify: `app/services/llm/service.py`
- Create: `app/services/query/__init__.py`
- Create: `app/services/query/orchestrator.py`

- [ ] **Step 1: Add `synthesize()` method to LLMService**

Add at the end of the `LLMService` class in `app/services/llm/service.py`:

```python
    def synthesize(
        self,
        query: str,
        chunks: list,
        language: str = "en",
    ) -> str:
        """
        Synthesize an answer from retrieved chunks.

        Args:
            query: User question.
            chunks: List of RetrievedChunk objects.
            language: Answer language.

        Returns:
            Synthesized answer text.
        """
        from app.prompts.new_flow import answer_synthesis as answer_prompts

        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            source = chunk.source if hasattr(chunk, "source") else chunk.get("source", {})
            doc_name = source.get("document_name", "Unknown")
            section = source.get("section", "")
            node_id = source.get("node_id", "")
            text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")

            context_parts.append(
                f"[Section {idx}]\n"
                f"Act: {doc_name}\n"
                f"Node: {node_id} — {section}\n"
                f"Content: {text}\n"
            )

        context = "\n\n".join(context_parts)
        act_name = "Finance Act"
        if chunks:
            first_source = chunks[0].source if hasattr(chunks[0], "source") else chunks[0].get("source", {})
            act_name = first_source.get("document_name", "Finance Act")

        system_prompt, user_prompt_template = answer_prompts.get_prompts(language)
        user_prompt = user_prompt_template.format(
            act_name=act_name,
            sections=context,
            query=query,
        )

        result = self.call(
            prompt=user_prompt,
            system_instruction=system_prompt,
            temperature=0.2,
            max_tokens=8192,
        )

        return result if isinstance(result, str) else result[0]
```

- [ ] **Step 2: Create QueryOrchestrator**

Create `app/services/query/__init__.py`:
```python
"""Query orchestration — thin layer between retrieval and LLM synthesis."""
```

Create `app/services/query/orchestrator.py`:

```python
"""
Query orchestrator — retrieves chunks via strategy, synthesizes via LLM.

Replaces the 647-line RAGOrchestrator with ~60 lines.
"""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.services.llm import LLMService
from app.services.retrieval.base import RetrievedChunk
from app.services.retrieval.factory import RetrievalFactory


@dataclass
class QueryResult:
    """Result returned by QueryOrchestrator.query()."""

    query: str
    answer: str
    chunks: list[RetrievedChunk]
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryOrchestrator:
    """Thin orchestrator: factory -> retrieve -> LLM synthesize."""

    def __init__(self, llm_service: LLMService | None = None):
        self.llm = llm_service or LLMService()

    async def query(
        self,
        user_query: str,
        top_k: int = 5,
        strategy_name: str | None = None,
        collection_name: str | None = None,
        filter_conditions: dict[str, Any] | None = None,
        use_llm: bool = True,
    ) -> QueryResult:
        logger.info(f"Query: {user_query[:100]!r}, strategy={strategy_name or 'default'}")

        strategy = await RetrievalFactory.get_strategy(strategy_name)

        chunks = await strategy.retrieve(
            query=user_query,
            top_k=top_k,
            filter_conditions=filter_conditions,
            collection_name=collection_name,
        )

        answer = ""
        if use_llm and chunks:
            answer = self.llm.synthesize(user_query, chunks)

        sources = [chunk.source for chunk in chunks]

        return QueryResult(
            query=user_query,
            answer=answer,
            chunks=chunks,
            sources=sources,
            metadata={
                "strategy": strategy_name or "default",
                "chunks_retrieved": len(chunks),
            },
        )
```

- [ ] **Step 3: Verify imports**

Run: `python -c "from app.services.query.orchestrator import QueryOrchestrator, QueryResult; print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add app/services/llm/service.py app/services/query/
git commit -m "feat: add QueryOrchestrator and LLMService.synthesize()"
```

---

## Task 10: Consolidate ingestion pipeline

**Files:**
- Create: `app/services/ingestion/pipeline.py`
- Create: `app/services/ingestion/processing.py`
- Copy: `app/services/processing/chunking.py` -> `app/services/ingestion/chunking.py`
- Modify: `app/services/ingestion/__init__.py`

- [ ] **Step 1: Copy chunking and create processing bridge**

```bash
cp app/services/processing/chunking.py app/services/ingestion/chunking.py
```

Create `app/services/ingestion/processing.py`:
```python
"""
Document processing — OCR to Markdown and metadata extraction.
Re-exports from existing modules during migration.
"""

from app.services.processing.markdown import DocumentProcessor
from app.services.processing.metadata import MetadataExtractor

__all__ = ["DocumentProcessor", "MetadataExtractor"]
```

- [ ] **Step 2: Create `pipeline.py`**

```python
"""
Ingestion pipeline — parse, process, store.

Single entry point for document ingestion. Strategy-agnostic:
delegates storage to whichever RetrievalStrategy is selected.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from app.services.retrieval.base import RetrievalStrategy


class IngestionPipeline:
    """Parse -> Process -> Store. One class, one flow."""

    def __init__(
        self,
        parser_factory=None,
        processor=None,
        metadata_extractor=None,
        file_storage=None,
        document_repo=None,
    ):
        self.parser_factory = parser_factory
        self.processor = processor
        self.metadata_extractor = metadata_extractor
        self.file_storage = file_storage
        self.document_repo = document_repo

    async def run(
        self,
        file_path: Path,
        filename: str,
        document_id: str,
        strategy: RetrievalStrategy,
        language: str = "en",
    ) -> dict[str, Any]:
        """Run the full ingestion pipeline."""
        logger.info(f"Ingesting document: {filename} (id={document_id})")

        try:
            # Step 1: Parse document (OCR)
            logger.debug("Step 1: Parsing document")
            parsed = await self.parser_factory.parse(str(file_path))

            # Step 2: Gather raw data and convert to Markdown
            logger.debug("Step 2: Gathering data and converting to Markdown")
            from app.services.ingestion.service import IngestionService
            from app.services.ingestion.processing import DocumentProcessor

            processor = self.processor or DocumentProcessor()
            ingestion_service = IngestionService()
            document_name = Path(filename).stem if filename else document_id

            gathered = ingestion_service.gather_document_data(
                parsed_response=parsed,
                document_name=document_name,
            )

            if gathered.get("status") != "success":
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": gathered.get("error", "Failed to gather data"),
                }

            markdown_result = processor.process_to_markdown(
                raw_ocr=gathered.get("raw_ocr", []),
                page_scalars=gathered.get("page_scalars", []),
                page_images=gathered.get("page_images", []),
            )
            markdown = markdown_result.get("markdown", "")

            if not markdown:
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": "No content extracted",
                }

            # Step 3: Extract metadata
            logger.debug("Step 3: Extracting metadata")
            from app.services.ingestion.processing import MetadataExtractor

            extractor = self.metadata_extractor or MetadataExtractor(language=language)
            metadata = extractor.extract_metadata(markdown, document_name=document_name)
            metadata["language"] = language
            metadata["document_name"] = document_name

            # Step 4: Store original PDF in GridFS
            if self.file_storage and str(file_path).lower().endswith(".pdf"):
                logger.debug("Step 4: Saving PDF to GridFS")
                try:
                    await self.file_storage.save_pdf(str(file_path), document_id, filename)
                except Exception as e:
                    logger.warning(f"Failed to save PDF: {e}")

            # Step 5: Strategy-specific storage
            logger.debug("Step 5: Strategy ingestion")
            await strategy.ingest(document_id, markdown, metadata)

            # Step 6: Record in document repository
            if self.document_repo:
                await self.document_repo.save(
                    document_id=document_id,
                    filename=filename,
                    metadata=metadata,
                    strategy=strategy.__class__.__name__,
                )

            logger.info(f"Ingestion complete: {document_id}")
            return {
                "document_id": document_id,
                "markdown": markdown,
                "metadata": metadata,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Ingestion failed for {document_id}: {e}")
            return {
                "document_id": document_id,
                "status": "failed",
                "error": str(e),
            }
```

- [ ] **Step 3: Update `app/services/ingestion/__init__.py`**

```python
"""Document ingestion — parse, process, store."""

from app.services.ingestion.pipeline import IngestionPipeline

__all__ = ["IngestionPipeline"]
```

- [ ] **Step 4: Verify import**

Run: `python -c "from app.services.ingestion.pipeline import IngestionPipeline; print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add app/services/ingestion/
git commit -m "feat: add consolidated IngestionPipeline with strategy-based storage"
```

---

## Task 11: Create DocumentRepository and rewrite documents endpoint

**Files:**
- Create: `app/db/repositories/document_repository.py`
- Rewrite: `app/api/v1/endpoints/documents.py`

- [ ] **Step 1: Create `document_repository.py`**

```python
"""
Document repository — consolidated document CRUD.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


class DocumentRepository:
    """CRUD for the documents collection."""

    def __init__(self, database):
        self.collection = database.documents

    async def save(
        self, document_id: str, filename: str, metadata: dict[str, Any], strategy: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "document_id": document_id,
                    "filename": filename,
                    "metadata": metadata,
                    "strategy": strategy,
                    "status": "completed",
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def get(self, document_id: str) -> Optional[dict]:
        return await self.collection.find_one({"document_id": document_id}, {"_id": 0})

    async def list_all(self, skip: int = 0, limit: int = 50) -> list[dict]:
        cursor = self.collection.find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit)
        return [doc async for doc in cursor]

    async def delete(self, document_id: str) -> bool:
        result = await self.collection.delete_one({"document_id": document_id})
        return result.deleted_count > 0
```

- [ ] **Step 2: Rewrite `documents.py` endpoint**

Replace `app/api/v1/endpoints/documents.py` with the full merged endpoint covering upload, list, get, pdf serve, and delete. See the documents.py code in the original plan (Task 11 of the full-test version). The key routes are:

- `POST /documents/upload` — upload PDF, run IngestionPipeline with selected strategy
- `GET /documents` — list all documents
- `GET /documents/{document_id}` — get document metadata
- `GET /documents/{document_id}/pdf` — serve PDF from GridFS
- `DELETE /documents/{document_id}` — delete document + cleanup

(Full code is in the spec — implement the merged endpoint as described in Design Section 3.)

- [ ] **Step 3: Commit**

```bash
git add app/db/repositories/document_repository.py app/api/v1/endpoints/documents.py
git commit -m "feat: add DocumentRepository and consolidated documents endpoint"
```

---

## Task 12: Create query endpoint and update router

**Files:**
- Create: `app/api/v1/endpoints/query.py`
- Modify: `app/api/v1/router.py`

- [ ] **Step 1: Create `query.py`**

```python
"""
Query endpoint — ask questions against ingested documents.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, status
from loguru import logger
from pydantic import BaseModel, Field

from app.services.query.orchestrator import QueryOrchestrator
from app.utils.auth import get_current_user
from app.utils.exceptions import AppException

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query/question", min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = Field(default=True)
    strategy: Optional[str] = Field(default=None, description="'page_index' or 'vector'")
    collection_name: Optional[str] = Field(default=None)
    filter_conditions: Optional[Dict[str, Any]] = Field(default=None)


class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks: list[Dict[str, Any]] = Field(default_factory=list)
    sources: list[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@router.post("", response_model=QueryResponse, summary="Ask a question")
async def query(
    request: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """Ask a question against ingested documents."""
    logger.info(f"Query: {request.query[:100]!r}")

    try:
        orchestrator = QueryOrchestrator()
        result = await orchestrator.query(
            user_query=request.query,
            top_k=request.top_k,
            strategy_name=request.strategy,
            collection_name=request.collection_name,
            filter_conditions=request.filter_conditions,
            use_llm=request.use_llm,
        )

        return QueryResponse(
            query=result.query,
            answer=result.answer,
            chunks=[
                {"text": c.text, "source": c.source, "score": c.score, "metadata": c.metadata}
                for c in result.chunks
            ],
            sources=result.sources,
            metadata=result.metadata,
        )

    except AppException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise AppException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="E_QUERY_FAILED",
            message=str(e),
        ) from e
```

- [ ] **Step 2: Update `router.py`**

Replace `app/api/v1/router.py`:

```python
"""
Main router for API v1 endpoints.
4 route groups: health, auth, documents, query.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import auth, documents, health, query

api_router = APIRouter()

# Public routes
api_router.include_router(health.router)
api_router.include_router(auth.router)

# Protected routes
api_router.include_router(documents.router)
api_router.include_router(query.router)
```

- [ ] **Step 3: Commit**

```bash
git add app/api/v1/endpoints/query.py app/api/v1/router.py
git commit -m "feat: add query endpoint and update router to 4 route groups"
```

---

## Task 13: Delete dead code and old files

Only run after all new code is in place.

- [ ] **Step 1: Delete old endpoint files**

```bash
rm app/api/v1/endpoints/ask.py
rm app/api/v1/endpoints/parse.py
rm app/api/v1/endpoints/collections.py
rm app/api/v1/endpoints/files.py
rm app/api/v1/endpoints/ingest.py
```

- [ ] **Step 2: Delete old service directories**

```bash
rm -rf app/services/rag/
rm -rf app/services/page_index/
rm -rf app/services/embeddings/
rm -rf app/services/vector_store/
rm -rf app/services/download/
```

- [ ] **Step 3: Delete dead retrieval files from old location**

```bash
rm -f app/services/retrieval/query_enhancer.py
rm -f app/services/retrieval/metadata_enhancer.py
rm -f app/services/retrieval/context_optimizer.py
rm -f app/services/retrieval/page_index_retriever.py
rm -f app/services/retrieval/hybrid_search.py
rm -f app/services/retrieval/bm25_search.py
rm -f app/services/retrieval/reranker.py
```

- [ ] **Step 4: Update `app/services/retrieval/__init__.py`**

```python
"""Retrieval strategies for RAG queries."""

from app.services.retrieval.base import RetrievalStrategy, RetrievedChunk
from app.services.retrieval.factory import RetrievalFactory

__all__ = ["RetrievalStrategy", "RetrievedChunk", "RetrievalFactory"]
```

- [ ] **Step 5: Delete old prompts**

```bash
rm -rf app/prompts/old_flow/
```

- [ ] **Step 6: Verify the app still imports cleanly**

Run: `python -c "from app.main import app; print('App imports OK')"`

Fix any remaining broken imports.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: remove dead code, old endpoints, and migrated service directories"
```

---

## Task 14: Integration smoke test

- [ ] **Step 1: Start the server**

Run: `python server.py`
Verify: No import errors, server starts.

- [ ] **Step 2: Check Swagger docs**

Open: `http://localhost:8000/docs`
Verify: 4 route groups visible (auth, health, documents, query).

- [ ] **Step 3: Test health endpoint**

Run: `curl http://localhost:8000/api/v1/health`
Verify: 200 OK.

- [ ] **Step 4: Test upload + query (if MongoDB available)**

Upload a test PDF, then query against it. Verify the full pipeline works end-to-end.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix: integration fixes from smoke test"
```
