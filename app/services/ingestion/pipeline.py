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
