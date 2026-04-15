"""
PageIndex retrieval strategy — vectorless RAG.

Implements RetrievalStrategy using LLM-guided hierarchical tree navigation.
"""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

from app.db.repositories.page_index_repository import PageIndexRepository
from app.services.llm import LLMService
from app.services.retrieval.base import RetrievalStrategy, RetrievedChunk
from app.services.retrieval.page_index.cache import (
    get_cached_markdown,
    get_cached_tree,
    invalidate_cache,
    save_to_cache,
)
from app.services.retrieval.page_index.section_retriever import SectionRetriever
from app.services.retrieval.page_index.tree_builder import TreeBuilder
from app.settings import settings


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
        if filter_conditions and "document_id" in filter_conditions:
            document_ids = [filter_conditions["document_id"]]
        else:
            document_ids = await self.repo.list_document_ids()
            if not document_ids:
                logger.warning("No PageIndex trees found")
                return []
            if collection_name:
                document_ids = await self._filter_by_collection(document_ids, collection_name)

        all_chunks: list[RetrievedChunk] = []

        for doc_id in document_ids:
            # Try file cache first, fall back to MongoDB
            tree_doc = get_cached_tree(doc_id)
            markdown = get_cached_markdown(doc_id)

            if not tree_doc or not markdown:
                tree_doc = await self.repo.get_tree(doc_id)
                markdown = await self.repo.get_markdown(doc_id)
                if not tree_doc or not markdown:
                    continue
                # Cache for next time
                save_to_cache(doc_id, tree_doc, markdown)


            tree = tree_doc.get("tree", {})
            language = tree_doc.get("language", "en")
            act_name = tree.get("document_title", doc_id)

            sections = await asyncio.to_thread(
                self.section_retriever.retrieve,
                query=query,
                tree=tree,
                markdown_content=markdown,
                language=language,
                top_k=top_k,
            )

            for idx, section in enumerate(sections):
                page_range = section.get("page_range", [])
                all_chunks.append(
                    RetrievedChunk(
                        text=section["text"],
                        source={
                            "document_name": act_name,
                            "document_id": doc_id,
                            "page_range": page_range,
                            "page": page_range[0] if page_range else None,
                            "section": section.get("title", ""),
                            "node_id": section["nodeId"],
                            "node_int_id": section.get("int_id"),
                            "page_bboxes": section.get("page_bboxes", []),
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
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
        page_bbox_map: list[dict] | None = None,
        image_dimensions: dict | None = None,
        custom_tree: dict | None = None,
    ) -> dict:
        # Invalidate file cache — will be rebuilt on first query
        invalidate_cache(document_id)
        language = metadata.get("language", "en")
        logger.info(f"PageIndex ingesting document_id={document_id}")

        async def _notify(msg: str) -> None:
            logger.info(f"[{document_id[:8]}] {msg}")
            if on_progress:
                try:
                    await on_progress(msg)
                except Exception as e:
                    logger.warning(f"Progress callback failed (non-fatal): {e}")

        if custom_tree is not None:
            await _notify("Using user-supplied page-index tree (skipping LLM generation)...")
            # deepcopy so we don't mutate the caller's dict (it gets written to
            # with start_char/end_char/id/page_bboxes by the post-processing steps).
            tree = deepcopy(custom_tree)
        else:
            # Split here so we can report async progress between chunks
            chunks = self.tree_builder.split_chunks(markdown)
            total = len(chunks)

            subtrees = []
            for i, chunk in enumerate(chunks, 1):
                await _notify(f"Building index tree: chunk {i}/{total} ({len(chunk):,} chars)...")
                # Run the blocking LLM call in a thread pool so the event loop
                # stays alive for MongoDB keepalives during the 1-2 min API call.
                subtree = await asyncio.to_thread(self.tree_builder.build_chunk, chunk, language)
                subtrees.append(subtree)

            if total == 1:
                tree = subtrees[0]
            else:
                tree = self.tree_builder._merge_trees(subtrees, language)

        tree["language"] = language
        unmatched_nodes = self.tree_builder._attach_char_offsets(
            tree.get("nodes", []), markdown, page_bbox_map
        )
        TreeBuilder._assign_node_ids(tree["nodes"])
        logger.info(f"[{document_id[:8]}] page_bbox_map received: {len(page_bbox_map) if page_bbox_map else 'None'} entries")
        if page_bbox_map:
            TreeBuilder._attach_page_bboxes(tree["nodes"], page_bbox_map)
            # Log first node's page_bboxes to verify
            if tree["nodes"]:
                first = tree["nodes"][0]
                logger.info(f"[{document_id[:8]}] First node page_bboxes: {first.get('page_bboxes', 'MISSING')}")

        await self.repo.save_tree(
            document_id=document_id,
            tree=tree,
            markdown=markdown,
            language=language,
            image_dimensions=image_dimensions,
        )
        node_count = self.tree_builder._count_nodes(tree.get("nodes", []))
        logger.info(f"PageIndex tree saved: {node_count} nodes")

        # Also dump the tree to a JSON file on disk for inspection/debugging.
        # Non-fatal: failure here must never break ingest.
        try:
            trees_dir = Path(settings.upload_dir) / "trees"
            trees_dir.mkdir(parents=True, exist_ok=True)
            tree_path = trees_dir / f"{document_id}.tree.json"
            tree_path.write_text(
                json.dumps(tree, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(f"[{document_id[:8]}] Tree written to {tree_path}")
        except Exception as e:  # noqa: BLE001 - disk dump is best-effort only
            logger.warning(f"[{document_id[:8]}] Failed to write tree.json (non-fatal): {e}")

        await _notify(f"PageIndex tree saved ({node_count} nodes)")

        # Build warnings payload (only meaningful for the custom-tree path —
        # LLM-generated trees produce titles drawn from the same markdown, so
        # unmatched_nodes there would just reflect OCR drift and isn't
        # actionable. We surface warnings only when a user supplied the tree.)
        if custom_tree is not None and unmatched_nodes:
            ingest_warnings = {
                "unmatched_nodes": unmatched_nodes,
                "unmatched_count": len(unmatched_nodes),
                "total_nodes": node_count,
            }
        else:
            ingest_warnings = None

        return {
            "custom_tree_provided": custom_tree is not None,
            "ingest_warnings": ingest_warnings,
            "node_count": node_count,
        }

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
