"""
Processing Service

Orchestrates all document processing operations:
1. OCR → Markdown conversion
2. Text chunking
3. Metadata extraction

This service handles all processing logic, separate from ingestion (which only gathers data).
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger

from app.services.processing.chunking import ChunkingService
from app.services.processing.markdown import DocumentProcessor
from app.services.processing.metadata import MetadataExtractor


class ProcessingService:
    """
    Main service for processing documents.

    This service orchestrates all processing operations:
    - Converts OCR to Markdown
    - Chunks the document
    - Extracts metadata
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        respect_sections: bool = True,
        generate_embeddings: bool = False,
        store_in_vector_db: bool = False,
    ):
        """
        Initialize the processing service.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Chunk overlap in characters
            respect_sections: Whether to respect section boundaries when chunking
            generate_embeddings: Whether to generate embeddings for chunks
            store_in_vector_db: Whether to store chunks in vector DB
        """
        self.processor = DocumentProcessor()
        self.chunking_service = ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_sections=respect_sections,
        )
        self.metadata_extractor = MetadataExtractor()
        
        # Optional: Embeddings and Vector DB
        self.generate_embeddings = generate_embeddings
        self.store_in_vector_db = store_in_vector_db
        
        if generate_embeddings or store_in_vector_db:
            from app.services.embeddings import EmbeddingService
            from app.services.retrieval import HybridSearchService
            from app.services.vector_store import VectorStoreService
            from app.settings import settings
            
            self.embedding_service = EmbeddingService() if generate_embeddings else None
            # Collection name will be determined dynamically from act_name
            self.vector_store = VectorStoreService() if store_in_vector_db else None
            
            # Initialize hybrid search service for BM25 indexing
            if store_in_vector_db and getattr(settings, "hybrid_search_enabled", True):
                from app.services.retrieval import BM25SearchService
                self.bm25_service = BM25SearchService()
                self.hybrid_search = HybridSearchService(
                    vector_store=self.vector_store,
                    bm25_service=self.bm25_service,
                    vector_weight=getattr(settings, "hybrid_search_vector_weight", 0.7),
                    bm25_weight=getattr(settings, "hybrid_search_bm25_weight", 0.3),
                )
            else:
                self.bm25_service = None
                self.hybrid_search = None

    def process_document(
        self,
        raw_ocr: List,
        page_scalars: Optional[List[Dict[str, Any]]] = None,
        page_images: Optional[List[Any]] = None,
        document_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a document through the entire processing pipeline.

        Args:
            raw_ocr: List of OCR DataFrames (one per page)
            page_scalars: Optional page metadata
            page_images: Optional page images for table extraction
            document_name: Optional document name/identifier

        Returns:
            Dictionary with processing results:
            {
                "document_id": str,
                "markdown": str,
                "chunks": List[Dict],
                "metadata": Dict,
                "tables": List[Dict],
                "fields": Dict,
                "status": str,
            }
        """
        logger.info(f"Processing document: {document_name or 'unnamed'}")

        # Step 1: Convert OCR to Markdown
        logger.debug("Step 1: Converting OCR to Markdown")
        markdown_result = self.processor.process_to_markdown(
            raw_ocr=raw_ocr,
            page_scalars=page_scalars,
            page_images=page_images,
        )

        markdown_content = markdown_result.get("markdown", "")
        if not markdown_content:
            logger.warning("No Markdown content generated")
            return {
                "document_id": document_name or "unknown",
                "markdown": "",
                "chunks": [],
                "metadata": {},
                "tables": [],
                "fields": {},
                "status": "failed",
                "error": "No content extracted",
            }

        # Step 2: Extract document-level metadata
        logger.debug("Step 2: Extracting metadata")
        # Pass document_name to help with act name extraction
        document_metadata = self.metadata_extractor.extract_metadata(
            markdown_content, document_name=document_name
        )
        if document_name:
            document_metadata["document_name"] = document_name
        
        # Log extracted act name for debugging
        if document_metadata.get("act_name"):
            logger.info(f"Extracted act name: {document_metadata['act_name']}")
        else:
            logger.warning(
                f"Could not extract act name from document. "
                f"Document name: {document_name or 'Not provided'}"
            )

        # Step 3: Chunk the document
        logger.debug("Step 3: Chunking document")
        chunks = self.chunking_service.chunk_document(markdown_content, document_metadata)

        # Step 4: Attach metadata to chunks
        logger.debug("Step 4: Enriching chunks with metadata")
        enriched_chunks = self.metadata_extractor.attach_metadata_to_chunks(chunks, document_metadata)

        document_id = document_name or "unknown"
        
        # Step 5: Generate embeddings (optional)
        embeddings = None
        if self.generate_embeddings and self.embedding_service:
            logger.debug("Step 5: Generating embeddings")
            try:
                chunk_texts = [chunk.get("text", "") for chunk in enriched_chunks]
                embeddings = self.embedding_service.generate_embeddings(chunk_texts)
                logger.info(f"Generated {len(embeddings)} embeddings")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings: {e}")
                embeddings = None

        # Step 6: Store in vector DB (optional)
        vector_store_status = None
        if self.store_in_vector_db and self.vector_store and embeddings:
            logger.debug("Step 6: Storing chunks in vector DB")
            try:
                # Determine collection name from act_name
                act_name = document_metadata.get("act_name")
                collection_name = self._get_collection_name(act_name)
                
                logger.info(f"Storing chunks in collection: {collection_name}")
                success = self.vector_store.store_chunks(
                    chunks=enriched_chunks,
                    embeddings=embeddings,
                    document_id=document_id,
                    collection_name=collection_name,
                )
                vector_store_status = "success" if success else "failed"
                logger.info(f"Vector DB storage: {vector_store_status} (collection: {collection_name})")
                
                # Build BM25 index for hybrid search
                if success and self.hybrid_search and self.bm25_service:
                    try:
                        logger.debug(f"Building BM25 index for collection: {collection_name}")
                        bm25_success = self.hybrid_search.build_bm25_index(
                            chunks=enriched_chunks,
                            collection_name=collection_name,
                        )
                        if bm25_success:
                            logger.info(f"BM25 index built successfully for: {collection_name}")
                        else:
                            logger.warning(f"Failed to build BM25 index for: {collection_name}")
                    except Exception as e:
                        logger.warning(f"Error building BM25 index: {e}")
            except Exception as e:
                logger.warning(f"Failed to store in vector DB: {e}")
                vector_store_status = "failed"

        result = {
            "document_id": document_id,
            "markdown": markdown_content,
            "chunks": enriched_chunks,
            "metadata": {
                **document_metadata,
                "total_chunks": len(enriched_chunks),
                "total_pages": markdown_result.get("metadata", {}).get("total_pages", 0),
                "total_words": markdown_result.get("metadata", {}).get("total_words", 0),
            },
            "tables": markdown_result.get("tables", []),
            "fields": markdown_result.get("fields", {}),
            "status": "success",
        }
        
        # Add embedding/vector DB status if applicable
        if embeddings is not None:
            result["embeddings_generated"] = len(embeddings)
        if vector_store_status:
            result["vector_store_status"] = vector_store_status

        logger.info(
            f"Processing complete: {len(enriched_chunks)} chunks, "
            f"Act: {document_metadata.get('act_name', 'Unknown')}"
        )

        return result

    @staticmethod
    def _get_collection_name(act_name: Optional[str]) -> str:
        """
        Convert act name to collection name.

        Args:
            act_name: Act name (e.g., "VAT Act", "Income Tax Act")

        Returns:
            Collection name (e.g., "vat_act", "income_tax_act")
        """
        if not act_name:
            return "unknown_act"

        # Normalize act name to collection name
        # Remove special characters, convert to lowercase, replace spaces with underscores
        collection_name = re.sub(r"[^a-zA-Z0-9\s]", "", act_name)
        collection_name = collection_name.lower().strip()
        collection_name = re.sub(r"\s+", "_", collection_name)
        
        # Ensure it's not empty
        if not collection_name:
            return "unknown_act"
        
        return collection_name
