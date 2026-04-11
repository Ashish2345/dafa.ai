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
