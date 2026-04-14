"""
Ingestion pipeline — parse, process, store.

Single entry point for document ingestion. Strategy-agnostic:
delegates storage to whichever RetrievalStrategy is selected.
"""

from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

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
        ocr_bbox_repo=None,
    ):
        self.parser_factory = parser_factory
        self.processor = processor
        self.metadata_extractor = metadata_extractor
        self.file_storage = file_storage
        self.document_repo = document_repo
        self.ocr_bbox_repo = ocr_bbox_repo

    async def run(
        self,
        file_path: Path,
        filename: str,
        document_id: str,
        strategy: RetrievalStrategy,
        language: str = "en",
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> dict[str, Any]:
        """Run the full ingestion pipeline.

        Args:
            on_progress: Optional async callback called with a human-readable step
                         description at each stage. Use this to persist live status.
        """
        logger.info(f"Ingesting document: {filename} (id={document_id})")

        async def _step(msg: str) -> None:
            logger.info(f"[{document_id[:8]}] {msg}")
            if on_progress:
                try:
                    await on_progress(msg)
                except Exception as _prog_err:
                    # Status update failures must never abort the ingestion
                    logger.warning(f"[{document_id[:8]}] Progress update failed (non-fatal): {_prog_err}")

        try:
            # Step 1: Parse document (OCR)
            await _step("Parsing document (OCR)...")
            parsed = await self.parser_factory.parse(str(file_path))

            # Step 2: Gather raw data and convert to Markdown
            await _step("Converting pages to Markdown...")
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
            page_bbox_map = markdown_result.get("page_bbox_map", [])
            # Persist word-level OCR bounding boxes (if repo available)
            word_bboxes = markdown_result.get("word_bboxes", [])
            if self.ocr_bbox_repo and word_bboxes:
                await _step("Saving word-level bounding boxes...")
                for page_entry in word_bboxes:
                    await self.ocr_bbox_repo.save_page(
                        document_id=document_id,
                        page=page_entry["page"],
                        words=page_entry["words"],
                    )
                logger.info(f"[{document_id[:8]}] Saved word bboxes for {len(word_bboxes)} pages")
            logger.info(f"[{document_id[:8]}] page_bbox_map has {len(page_bbox_map)} entries"
                        + (f", first: page={page_bbox_map[0]['page']}, bbox={page_bbox_map[0]['bbox']}" if page_bbox_map else ""))

            image_dimensions = None
            if gathered.get("page_scalars") and len(gathered["page_scalars"]) > 0:
                first_page = gathered["page_scalars"][0]
                image_dimensions = {
                    "width": first_page.get("width", 0),
                    "height": first_page.get("height", 0),
                }

            if not markdown:
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": "No content extracted",
                }

            # Step 3: Extract metadata
            await _step("Extracting metadata...")
            from app.services.ingestion.processing import MetadataExtractor

            extractor = self.metadata_extractor or MetadataExtractor(language=language)
            metadata = extractor.extract_metadata(markdown, document_name=document_name)
            metadata["language"] = language
            metadata["document_name"] = document_name

            # Step 4: Store original PDF and page images in GridFS
            if self.file_storage and str(file_path).lower().endswith(".pdf"):
                await _step("Saving PDF and page images to storage...")
                try:
                    await self.file_storage.save_pdf(str(file_path), document_id, filename)
                    # Convert PDF pages to images using PyMuPDF and save to GridFS
                    import fitz
                    doc = fitz.open(str(file_path))
                    scale = 150 / 72  # 150 DPI
                    mat = fitz.Matrix(scale, scale)
                    for page_num in range(len(doc)):
                        pix = doc[page_num].get_pixmap(matrix=mat)
                        img_bytes = pix.tobytes("jpeg")
                        await self.file_storage.save_image(
                            image_data=img_bytes,
                            document_id=document_id,
                            page_number=page_num + 1,
                        )
                    logger.info(f"Saved {len(doc)} page images for {document_id}")
                    doc.close()
                except Exception as e:
                    logger.warning(f"Failed to save PDF/images: {e}")

            # Step 5: Strategy-specific storage (passes on_progress so chunked
            # strategies can report per-chunk status)
            await strategy.ingest(
                document_id,
                markdown,
                metadata,
                on_progress=on_progress,
                page_bbox_map=page_bbox_map,
                image_dimensions=image_dimensions,
            )

            # Step 6: Record in document repository
            await _step("Saving document record...")
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
