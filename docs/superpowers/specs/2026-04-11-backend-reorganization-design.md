# Backend Reorganization Design

**Date:** 2026-04-11
**Status:** Approved
**Scope:** Restructure backend around strategy pattern for dual RAG approaches, consolidate routes, remove dead code, fix PageIndex bugs.

---

## Context

The dafa.ai backend is a FastAPI service powering Smart Finance Compliance — a RAG-based platform for Nepal's financial/legal compliance market. It ingests scanned Nepali government PDFs (Acts, NRB directives, IRD circulars) and answers compliance questions with cited sources.

The backend currently has two retrieval approaches tangled together:
- **PageIndex (vectorless RAG):** LLM builds a hierarchical tree from Markdown, then navigates it at query time. No embeddings needed.
- **Vector RAG:** Traditional embeddings + Qdrant vector search + BM25 hybrid search + cross-encoder re-ranking.

Problems:
- Both approaches are mixed into a single 647-line `RAGOrchestrator`
- Switching is a hard-coded `if settings.use_page_index` in the route handler
- Vector RAG code is scattered across 7 service directories
- 8 flat API endpoint files with overlap and dead routes
- Dead code: `QueryEnhancer` results unused, `parse` endpoint redundant, `CollectionRouter` overkill
- Hardcoded MongoDB connection string overriding settings
- PageIndex has fragile regex-based text extraction

---

## Design Decisions

1. **Both strategies remain fully active and switchable per-request** via a `strategy` field in the request body, defaulting to `"page_index"` from settings.
2. **Strategy Pattern (Approach 1):** Common `RetrievalStrategy` interface with two implementations in separate folders, selected by a factory.
3. **4-route API model:** `auth`, `health`, `documents` (upload/ingest/list/get/files), `query` (ask questions).
4. **No frontend constraints** — routes can be restructured freely.

---

## Target Project Structure

```
app/
  api/
    v1/
      router.py                      # 4 route groups
      endpoints/
        auth.py                      # Unchanged
        health.py                    # Unchanged
        documents.py                 # Merged: upload, ingest, list, get, files, delete
        query.py                     # Renamed from ask.py

  models/
    enums.py                         # Add RetrievalStrategy enum
    schemas.py                       # Slim down
    domain.py                        # Keep
    user.py                          # Keep

  db/
    mongodb.py                       # Fix hardcoded connection string
    repositories/
      document_repository.py         # New: consolidated document queries
      file_storage.py                # Keep (GridFS)
      page_index_repository.py       # Keep
      user_repository.py             # Keep

  services/
    retrieval/                       # Strategy pattern home
      base.py                        # RetrievalStrategy ABC
      factory.py                     # Picks strategy from request/settings
      page_index/
        strategy.py                  # Implements RetrievalStrategy
        tree_builder.py              # Build tree from markdown via LLM
        section_retriever.py         # Navigate tree + extract text
      vector/
        strategy.py                  # Implements RetrievalStrategy
        hybrid_search.py             # Vector + BM25 combined
        bm25_search.py               # BM25 keyword search
        reranker.py                  # Cross-encoder re-ranking
        embeddings.py                # OpenAI embedding service
        vector_store.py              # Qdrant client

    ingestion/
      pipeline.py                    # Single orchestrator: parse -> process -> store
      processing.py                  # Markdown conversion + metadata extraction
      chunking.py                    # Legal-aware chunking (used by vector strategy)

    llm/
      service.py                     # Keep + add synthesize() method

    parsers/                         # Keep as-is
      factory.py
      base.py
      pdf/
      image/

    query/
      orchestrator.py                # Thin: factory -> retrieve -> LLM synthesize

  config/                            # Keep
  prompts/                           # Clean up old_flow/
  utils/                             # Keep
  settings.py                        # Clean up dead feature flags
```

### Deleted

| Path | Reason |
|------|--------|
| `services/rag/` | Replaced by `services/query/` + `services/retrieval/` |
| `services/page_index/` | Moves into `services/retrieval/page_index/` |
| `services/embeddings/` | Moves into `services/retrieval/vector/embeddings.py` |
| `services/vector_store/` | Moves into `services/retrieval/vector/vector_store.py` |
| `services/processing/` | Consolidates into `services/ingestion/` |
| `services/download/` | Folds into route handler |
| `services/ingestion/service.py` | Replaced by `ingestion/pipeline.py` |
| `services/retrieval/query_enhancer.py` | Dead code (results logged but never used) |
| `services/retrieval/metadata_enhancer.py` | Dead feature — metadata filtering moves into strategy |
| `services/retrieval/context_optimizer.py` | 10 lines inlined into query orchestrator |
| `services/rag/collection_router.py` | Overkill LLM-based routing removed |
| `api/v1/endpoints/parse.py` | Redundant with ingest |
| `api/v1/endpoints/ask.py` | Renamed to `query.py` |
| `api/v1/endpoints/collections.py` | Merged into `documents.py` |
| `api/v1/endpoints/files.py` | Merged into `documents.py` |
| `api/v1/endpoints/ingest.py` | Merged into `documents.py` |
| `prompts/old_flow/` | Replaced by strategy-specific prompts |

---

## Retrieval Strategy Interface

```python
# app/services/retrieval/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    """Common output shape for all retrieval strategies."""
    text: str
    source: dict        # document_name, page_range, section
    score: float
    metadata: dict

class RetrievalStrategy(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: dict | None = None,
        collection_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return ranked chunks relevant to the query."""
        ...

    @abstractmethod
    async def ingest(
        self,
        document_id: str,
        markdown: str,
        metadata: dict,
    ) -> None:
        """Store a processed document for later retrieval."""
        ...
```

Both `retrieve` and `ingest` are on the same interface because storage format is strategy-specific:
- **PageIndex:** `ingest()` builds LLM tree, stores tree + markdown in MongoDB.
- **Vector:** `ingest()` chunks, embeds, stores in Qdrant.

---

## Retrieval Factory

```python
# app/services/retrieval/factory.py

class RetrievalFactory:
    @staticmethod
    async def get_strategy(strategy_name: str | None = None) -> RetrievalStrategy:
        name = strategy_name or settings.default_retrieval_strategy  # "page_index"
        if name == "page_index":
            return PageIndexStrategy(db=..., llm=...)
        elif name == "vector":
            return VectorStrategy(qdrant=..., embeddings=...)
        raise ValueError(f"Unknown strategy: {name}")
```

---

## Route Design

### `POST /api/v1/documents/upload`

Upload PDF, run full ingestion pipeline. Accepts optional `strategy` param (default: `page_index`).

### `GET /api/v1/documents`

List all ingested documents from MongoDB (strategy-agnostic).

### `GET /api/v1/documents/{document_id}`

Get document metadata + status.

### `GET /api/v1/documents/{document_id}/pdf`

Serve original PDF from GridFS.

### `GET /api/v1/documents/{document_id}/images/{page}`

Serve page image from GridFS.

### `DELETE /api/v1/documents/{document_id}`

Delete document + its strategy-specific data.

### `POST /api/v1/query`

Ask a question. Body: `query`, `top_k`, `strategy` (optional override), `collection_name` (optional), `filter_conditions` (optional), `use_llm` (default true).

---

## Ingestion Pipeline

```python
# app/services/ingestion/pipeline.py

class IngestionPipeline:
    async def run(
        self,
        file_path: Path,
        filename: str,
        document_id: str,
        strategy: RetrievalStrategy,
    ) -> IngestionResult:
        # 1. Parse: PDF/image -> raw OCR
        parsed = self.parser_factory.get_parser(file_path).parse()

        # 2. Process: OCR -> Markdown + metadata
        markdown = self.processor.to_markdown(parsed)
        metadata = self.metadata_extractor.extract(filename, markdown)

        # 3. Store original files in GridFS
        await self.file_storage.save_pdf(file_path, document_id)

        # 4. Strategy-specific storage
        await strategy.ingest(document_id, markdown, metadata)

        # 5. Record in MongoDB
        await self.document_repo.save(document_id, filename, metadata, strategy_name)

        return IngestionResult(document_id=document_id, status="completed", metadata=metadata)
```

The pipeline is strategy-agnostic. It parses and processes the document, then delegates storage to whichever strategy was selected. `scripts/inject.py` batch CLI continues to work by instantiating the pipeline and strategy directly.

---

## Query Orchestrator

```python
# app/services/query/orchestrator.py

class QueryOrchestrator:
    async def query(
        self,
        user_query: str,
        top_k: int = 5,
        strategy_name: str | None = None,
        collection_name: str | None = None,
        filter_conditions: dict | None = None,
        use_llm: bool = True,
    ) -> QueryResult:
        # 1. Get the right strategy
        strategy = await RetrievalFactory.get_strategy(strategy_name)

        # 2. Retrieve chunks
        chunks = await strategy.retrieve(
            query=user_query,
            top_k=top_k,
            filter_conditions=filter_conditions,
            collection_name=collection_name,
        )

        # 3. Synthesize answer (or return chunks only)
        if use_llm and chunks:
            answer = await self.llm.synthesize(user_query, chunks)
        else:
            answer = ""

        return QueryResult(
            query=user_query,
            answer=answer,
            chunks=chunks,
            sources=[chunk.source for chunk in chunks],
        )
```

~50 lines. All retrieval complexity lives inside the strategy. All LLM prompt logic lives in `llm/service.py`. The orchestrator is just glue.

---

## PageIndex Fixes

Applied during the move into `services/retrieval/page_index/`:

1. **Node text extraction:** Replace fragile regex-based heading search with character offsets (`start_char`, `end_char`) stored during tree building. Extraction becomes `markdown[start:end]`.

2. **Duplicate deduplication:** Deduplicate nodeIds after LLM navigation, before text extraction.

3. **Cross-document top-k:** Use relevance scores from LLM navigator to rank globally across documents before applying `top_k`.

4. **Large document handling:** Split at natural boundaries (chapter/section breaks) if over 800K chars, build tree per part, merge trees.

---

## Settings Cleanup

**Removed from top-level settings** (become strategy internals):
- `hybrid_search_enabled`, `hybrid_search_vector_weight`, `hybrid_search_bm25_weight`
- `rerank_use_llm`
- `metadata_enhancement_enabled`, `metadata_boost_weight`
- `context_optimization_enabled`, `context_reserved_tokens`
- `use_page_index` (replaced by `default_retrieval_strategy`)

**Added:**
- `default_retrieval_strategy: str = "page_index"`

**Fixed:**
- `mongodb.py` hardcoded connection string replaced with `settings.mongodb_url`

---

## Error Handling

- Strategy-specific errors (Qdrant down, tree navigation failed) are caught inside each strategy and wrapped in a common `RetrievalError` exception.
- The query orchestrator catches `RetrievalError` and returns a structured error response.
- The ingestion pipeline catches parse/process/store failures independently and records partial status in MongoDB (e.g., "parsed but storage failed").

---

## Testing Strategy

- **Unit tests per strategy:** Each strategy is independently testable with mocked dependencies (mock LLM for PageIndex tree building, mock Qdrant for vector).
- **Integration test for pipeline:** Upload a small PDF, run through pipeline, verify storage in MongoDB/GridFS.
- **Integration test for query:** Ingest a known document, query it, verify chunks are returned.
- **Strategy factory test:** Verify correct strategy is returned for each name, verify default fallback.
