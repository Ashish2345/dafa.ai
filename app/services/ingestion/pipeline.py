"""
Ingestion pipeline — parse, process, store.

Single entry point for document ingestion. Strategy-agnostic:
delegates storage to whichever RetrievalStrategy is selected.
"""

import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

from app.services.retrieval.base import RetrievalStrategy


_LINE_NUMBER_MARKER_RE = re.compile(r"->\s*\d+\s*<-\s*")
_PAGE_HEADER_RE = re.compile(r"^##\s*Page\s+(\d+)\s*$", re.MULTILINE)


def _strip_line_markers(text: str) -> str:
    """Scrub ``-> N<-`` line-number artifacts that leak out of the OCR parser.

    Tolerant of whitespace padding inside the delimiters — the OCR parser
    right-justifies the number (so lines look like ``-> 6<-``, ``->  12<-``).
    """
    return _LINE_NUMBER_MARKER_RE.sub("", text)


def _recompute_word_char_offsets(markdown: str, word_bboxes: list[dict]) -> None:
    """Re-index every word's ``char_offset`` into ``markdown``.

    Mutates ``word_bboxes`` in place. Used after stripping line-number
    markers from an uploaded parsed dump: the markdown just shortened, so
    the supplied offsets (computed against the dirty text) no longer point
    at real word positions.

    Strategy per page:
      * Find the page's slice by locating its ``## Page N`` header.
      * Walk the words in the order they arrived, using a forward-only
        cursor to disambiguate repeated tokens.
      * Case-insensitive fallback when the literal text is not found.
      * Words that can't be located fall back to the cursor tip (best effort
        — better than a stale offset pointing past end-of-string).
    """
    # Build page_number → (start_char, end_char) over the cleaned markdown.
    page_slices: dict[int, tuple[int, int]] = {}
    matches = list(_PAGE_HEADER_RE.finditer(markdown))
    for idx, m in enumerate(matches):
        page_no = int(m.group(1))
        slice_start = m.end()
        slice_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        page_slices[page_no] = (slice_start, slice_end)

    for page_entry in word_bboxes:
        page_no = page_entry.get("page")
        words = page_entry.get("words", []) or []
        if page_no is None or not words:
            continue

        slice_start, slice_end = page_slices.get(page_no, (0, len(markdown)))
        page_md = markdown[slice_start:slice_end]
        page_md_lower = page_md.lower()
        cursor = 0

        for w in words:
            text = str(w.get("text", "") or "").strip()
            if not text:
                w["char_offset"] = slice_start + cursor
                continue

            pos = page_md.find(text, cursor)
            if pos == -1:
                pos = page_md_lower.find(text.lower(), cursor)
            if pos >= 0:
                w["char_offset"] = slice_start + pos
                cursor = pos + len(text)
            else:
                # Word not found — peg to the cursor tip so highlighting
                # still falls roughly in the right area instead of way past
                # the end of the markdown (the pre-fix state).
                w["char_offset"] = slice_start + cursor


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
        custom_tree: dict | None = None,
        title: str | None = None,
        user_summary: str | None = None,
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

            # Step 3b: ensure we have a one-line summary on the document record.
            from app.services.ingestion.summary import SummaryGenerator

            effective_summary = (user_summary or "").strip()
            if not effective_summary:
                await _step("Generating summary...")
                gen = SummaryGenerator()
                effective_summary = await gen.generate(
                    title=title or document_name,
                    markdown=markdown,
                    language=language,
                )

            # Persist the summary onto the stub document record so the classifier
            # catalog picks it up as soon as the doc moves to 'completed'.
            if self.document_repo:
                await self.document_repo.set_summary(document_id, effective_summary)

            # Step 4: Store original PDF and page images in GridFS
            await self._save_pdf_and_images(
                file_path=str(file_path),
                document_id=document_id,
                filename=filename,
                step=_step,
            )

            # Step 5: Strategy-specific storage (passes on_progress so chunked
            # strategies can report per-chunk status).
            # Only PageIndexStrategy accepts custom_tree; VectorStrategy does not.
            # Omit the kwarg when None so we stay compatible with strategies
            # that don't expose it.
            ingest_kwargs: dict[str, Any] = {
                "on_progress": on_progress,
                "page_bbox_map": page_bbox_map,
                "image_dimensions": image_dimensions,
            }
            if custom_tree is not None:
                ingest_kwargs["custom_tree"] = custom_tree

            strategy_result = await strategy.ingest(
                document_id,
                markdown,
                metadata,
                **ingest_kwargs,
            )
            # Back-compat: VectorStrategy.ingest() currently returns None.
            if not isinstance(strategy_result, dict):
                strategy_result = {}

            # Step 6: Record in document repository
            await _step("Saving document record...")
            if self.document_repo:
                await self.document_repo.save(
                    document_id=document_id,
                    filename=filename,
                    metadata=metadata,
                    strategy=strategy.__class__.__name__,
                    custom_tree_provided=bool(strategy_result.get("custom_tree_provided", False)),
                    ingest_warnings=strategy_result.get("ingest_warnings"),
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

    async def _save_pdf_and_images(
        self,
        *,
        file_path: str,
        document_id: str,
        filename: str,
        step: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> Optional[dict]:
        """Persist the original PDF + a rendered image per page to GridFS.

        Best-effort: exceptions are logged and swallowed so ingestion flow
        stays resilient to storage hiccups. No-op when ``file_storage`` is
        not wired or the file isn't a PDF.

        Returns:
            ``{"width": int, "height": int}`` of the rendered page 1 image
            (the frontend multiplies normalized 0-1 bboxes by these), or
            ``None`` when no PDF was rendered.
        """
        if not self.file_storage or not file_path.lower().endswith(".pdf"):
            return None
        if step:
            await step("Saving PDF and page images to storage...")
        image_dimensions: Optional[dict] = None
        try:
            await self.file_storage.save_pdf(file_path, document_id, filename)
            import fitz

            doc = fitz.open(file_path)
            scale = 150 / 72  # 150 DPI
            mat = fitz.Matrix(scale, scale)
            for page_num in range(len(doc)):
                pix = doc[page_num].get_pixmap(matrix=mat)
                if image_dimensions is None:
                    image_dimensions = {"width": pix.width, "height": pix.height}
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
        return image_dimensions

    async def run_from_parsed_content(
        self,
        *,
        document_id: str,
        filename: str,
        markdown: str,
        word_bboxes: list[dict],
        strategy: RetrievalStrategy,
        language: str = "en",
        on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
        custom_tree: Optional[dict] = None,
        pdf_path: Optional[str] = None,
        title: str | None = None,
        user_summary: str | None = None,
    ) -> dict[str, Any]:
        """Ingest from pre-parsed content (OCR + markdown already done by caller).

        Skips the OCR + markdown-conversion stages entirely. Used when the
        client uploads the JSON dump produced by ``POST /documents/parse``.

        Args:
            markdown: Full markdown body (becomes ``page_index_content.markdown``).
            word_bboxes: List of ``{"page": int, "words": [...]}`` dicts — each
                entry persisted to ``page_ocr_bboxes`` as-is. Words must carry
                0-1 normalized ``x0/y0/x2/y2`` and a ``char_offset`` into the
                markdown.
        """
        logger.info(f"Ingesting from parsed content: {filename} (id={document_id})")

        async def _step(msg: str) -> None:
            logger.info(f"[{document_id[:8]}] {msg}")
            if on_progress:
                try:
                    await on_progress(msg)
                except Exception as _prog_err:
                    logger.warning(f"[{document_id[:8]}] Progress update failed (non-fatal): {_prog_err}")

        try:
            if not markdown:
                return {
                    "document_id": document_id,
                    "status": "failed",
                    "error": "Empty markdown supplied",
                }

            # Defensive cleanup: the uploaded parsed JSON may have been
            # produced before the line-marker regex fix, in which case the
            # markdown still contains ``-> N<-`` artifacts AND the word
            # ``char_offset`` values are indexed into that dirty markdown.
            # Strip markers AND re-compute every word's offset against the
            # cleaned text — otherwise the highlight endpoint queries the
            # right node range but finds zero words (offsets shifted).
            dirty_len = len(markdown)
            markdown = _strip_line_markers(markdown)
            if len(markdown) != dirty_len:
                logger.info(
                    f"[{document_id[:8]}] Stripped line-number markers "
                    f"from uploaded markdown ({dirty_len} -> {len(markdown)} chars)"
                )
                _recompute_word_char_offsets(markdown, word_bboxes)

            # Persist original PDF + rendered page images when provided —
            # the caller owns cleanup of the temp file. Capture the page 1
            # pixel dimensions so the highlight endpoint can scale 0-1
            # normalized bboxes back to image pixels.
            image_dimensions: Optional[dict] = None
            if pdf_path:
                image_dimensions = await self._save_pdf_and_images(
                    file_path=pdf_path,
                    document_id=document_id,
                    filename=filename,
                    step=_step,
                )

            # Persist word-level bboxes per page.
            if self.ocr_bbox_repo and word_bboxes:
                await _step("Saving word-level bounding boxes...")
                for page_entry in word_bboxes:
                    await self.ocr_bbox_repo.save_page(
                        document_id=document_id,
                        page=page_entry["page"],
                        words=page_entry.get("words", []),
                    )
                logger.info(
                    f"[{document_id[:8]}] Saved word bboxes for {len(word_bboxes)} pages"
                )

            # Build the page_bbox_map the strategy needs (page-level char ranges
            # + bounding boxes). Word offsets are already embedded in the
            # parsed payload so we just aggregate per page.
            page_bbox_map = _page_bbox_map_from_words(word_bboxes, len(markdown))

            await _step("Extracting metadata...")
            from app.services.ingestion.processing import MetadataExtractor

            document_name = Path(filename).stem if filename else document_id
            extractor = self.metadata_extractor or MetadataExtractor(language=language)
            metadata = extractor.extract_metadata(markdown, document_name=document_name)
            metadata["language"] = language
            metadata["document_name"] = document_name

            from app.services.ingestion.summary import SummaryGenerator

            effective_summary = (user_summary or "").strip()
            if not effective_summary:
                await _step("Generating summary...")
                gen = SummaryGenerator()
                effective_summary = await gen.generate(
                    title=title or document_name,
                    markdown=markdown,
                    language=language,
                )

            if self.document_repo:
                await self.document_repo.set_summary(document_id, effective_summary)

            ingest_kwargs: dict[str, Any] = {
                "on_progress": on_progress,
                "page_bbox_map": page_bbox_map,
                "image_dimensions": image_dimensions,
            }
            if custom_tree is not None:
                ingest_kwargs["custom_tree"] = custom_tree

            strategy_result = await strategy.ingest(
                document_id, markdown, metadata, **ingest_kwargs,
            )
            if not isinstance(strategy_result, dict):
                strategy_result = {}

            await _step("Saving document record...")
            if self.document_repo:
                await self.document_repo.save(
                    document_id=document_id,
                    filename=filename,
                    metadata=metadata,
                    strategy=strategy.__class__.__name__,
                    custom_tree_provided=bool(strategy_result.get("custom_tree_provided", False)),
                    ingest_warnings=strategy_result.get("ingest_warnings"),
                )

            logger.info(f"Ingestion (from parsed content) complete: {document_id}")
            return {
                "document_id": document_id,
                "markdown": markdown,
                "metadata": metadata,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Ingestion (from parsed content) failed for {document_id}: {e}")
            return {
                "document_id": document_id,
                "status": "failed",
                "error": str(e),
            }


def _page_bbox_map_from_words(word_bboxes: list[dict], markdown_len: int) -> list[dict]:
    """Derive the ``page_bbox_map`` from per-page word data.

    Each entry:
      * ``page``        — 1-indexed page number.
      * ``start_char`` / ``end_char`` — range of word char offsets on that page.
      * ``bbox``        — enclosing bbox of all words on the page (0-1 coords).
                          Falls back to a full-page bbox when a page has zero
                          words — keeps downstream consumers happy without
                          rendering phantom highlights.
    """
    out: list[dict] = []
    for entry in word_bboxes:
        page = entry.get("page")
        words = entry.get("words", []) or []
        if page is None:
            continue

        if words:
            offsets = [w.get("char_offset", 0) for w in words if "char_offset" in w]
            start_char = min(offsets) if offsets else 0
            end_char = max(
                (w.get("char_offset", 0) + len(str(w.get("text", ""))))
                for w in words
            )
            x0 = min(float(w.get("x0", 0.0)) for w in words)
            y0 = min(float(w.get("y0", 0.0)) for w in words)
            x2 = max(float(w.get("x2", 1.0)) for w in words)
            y2 = max(float(w.get("y2", 1.0)) for w in words)
        else:
            start_char = 0
            end_char = markdown_len
            x0, y0, x2, y2 = 0.0, 0.0, 1.0, 1.0

        out.append({
            "page": page,
            "start_char": start_char,
            "end_char": end_char,
            "bbox": {"x0": x0, "y0": y0, "x2": x2, "y2": y2},
        })
    out.sort(key=lambda e: e["page"])
    return out

