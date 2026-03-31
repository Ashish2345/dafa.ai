"""Batch injection orchestrator: discover → dedupe → fetch → parse → RAG storage."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.api.v1.forms import ParseFormData
from app.config.request_mapping import RequestConfigBuilder
from app.db.mongodb import mongodb
from app.services.ingestion import IngestionService
from app.services.injection.registry import SourceRegistry
from app.services.injection.sources.base import BaseDocumentSource, DocumentInfo
from app.services.injection.tracker import InjectionTracker, LocalInjectionTracker
from app.services.parsers.factory import ParserFactory
from app.services.pdf_storage import PDFStorageService
from app.services.processing import ProcessingService
from app.settings import Settings


@dataclass
class InjectionReport:
    discovered: int = 0
    skipped: int = 0
    processed: int = 0
    failed: int = 0
    dry_run: bool = False
    download_only: bool = False
    errors: List[str] = field(default_factory=list)
    discovered_items: List[Dict[str, Any]] = field(default_factory=list)
    downloaded_items: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "discovered": self.discovered,
            "skipped": self.skipped,
            "processed": self.processed,
            "failed": self.failed,
            "dry_run": self.dry_run,
            "download_only": self.download_only,
            "errors": list(self.errors),
        }
        if self.discovered_items:
            d["discovered_items"] = self.discovered_items
        if self.downloaded_items:
            d["downloaded_items"] = self.downloaded_items
        return d


class BatchInjectionOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._registry = SourceRegistry(settings)

    async def run(
        self,
        categories: Optional[List[str]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        force: bool = False,
        dry_run: bool = False,
        concurrency: Optional[int] = None,
        tracker: Optional[InjectionTracker | LocalInjectionTracker] = None,
        download_only_dir: Optional[Path] = None,
    ) -> InjectionReport:
        report = InjectionReport(dry_run=dry_run, download_only=download_only_dir is not None)
        if tracker is None:
            db = mongodb.get_database()
            tracker = InjectionTracker(db)

        if categories is None:
            category_filter = None
        else:
            category_filter = list(categories)

        sources = self._registry.build_sources(category_filter=category_filter)
        if not sources:
            report.errors.append("No enabled sources match the category filter.")
            return report

        all_docs: List[tuple[str, DocumentInfo]] = []
        for cat_key, source in sources:
            try:
                found = await source.discover(year_from=year_from, year_to=year_to)
                for d in found:
                    all_docs.append((cat_key, d))
            except Exception as e:
                msg = f"Discover failed for category {cat_key}: {e}"
                logger.exception(msg)
                report.errors.append(msg)

        report.discovered = len(all_docs)

        if dry_run:
            logger.info("Dry run: {} document(s) discovered", report.discovered)
            for cat_key, doc in all_docs:
                logger.info(
                    "  [{}] {} | year={} | source_id={}…",
                    cat_key,
                    doc.title,
                    doc.year,
                    doc.source_id[:8],
                )
            report.discovered_items = [
                {
                    "source_key": cat_key,
                    "title": doc.title,
                    "year": doc.year,
                    "url": doc.url,
                    "local_path": doc.local_path,
                }
                for cat_key, doc in all_docs
            ]
            return report

        conc = concurrency if concurrency is not None else self._settings.injection_concurrency
        sem = asyncio.Semaphore(max(1, conc))
        report_lock = asyncio.Lock()

        async def handle_one(cat_key: str, doc: DocumentInfo) -> None:
            async with sem:
                if not force and await tracker.is_ingested(doc.source_id):
                    async with report_lock:
                        report.skipped += 1
                    return
                if force:
                    await tracker.delete_by_source_id(doc.source_id)

                file_path: Optional[str] = None
                try:
                    await tracker.mark_in_progress(
                        doc.source_id,
                        title=doc.title,
                        category=doc.category,
                        year=doc.year,
                    )
                    source = self._source_for_category(sources, cat_key)
                    file_path = await source.fetch(doc)
                    if download_only_dir is not None:
                        from app.services.injection.local_save import (
                            act_pdf_destination,
                            copy_fetched_pdf_to_download_dir,
                            finance_acts_storage_info,
                        )

                        dest = act_pdf_destination(download_only_dir, cat_key, doc)
                        copy_fetched_pdf_to_download_dir(file_path, dest)
                        result = {
                            "document_id": f"local-{doc.source_id[:12]}",
                            "chunks": [],
                            "metadata": doc.metadata or {},
                            "vector_store_status": None,
                            "collection_name": None,
                            "saved_to": str(dest),
                        }
                        if cat_key == "finance_acts":
                            result["finance_storage"] = finance_acts_storage_info(doc)
                        await tracker.mark_ingested(
                            doc.source_id,
                            document_id=str(result["document_id"]),
                            title=doc.title,
                            category=doc.category,
                            year=doc.year,
                            result=result,
                        )
                        async with report_lock:
                            report.processed += 1
                            dl_entry: Dict[str, Any] = {
                                "source_key": cat_key,
                                "title": doc.title,
                                "year": doc.year,
                                "path": str(dest),
                            }
                            if cat_key == "finance_acts":
                                dl_entry["finance_storage"] = finance_acts_storage_info(doc)
                                try:
                                    dl_entry["relative_to_download_dir"] = str(
                                        dest.relative_to(download_only_dir.resolve())
                                    )
                                except ValueError:
                                    dl_entry["relative_to_download_dir"] = str(dest)
                            report.downloaded_items.append(dl_entry)
                    else:
                        result = await self._ingest_file(
                            file_path=file_path,
                            document_name=self._safe_document_name(doc),
                            original_filename=self._original_filename(doc, file_path),
                        )
                        doc_uuid = result.get("document_id")
                        if not doc_uuid:
                            raise RuntimeError("Ingestion returned no document_id")
                        qdrant_collection = ProcessingService._get_collection_name(
                            (result.get("metadata") or {}).get("act_name")
                        )
                        result["collection_name"] = qdrant_collection
                        await tracker.mark_ingested(
                            doc.source_id,
                            document_id=str(doc_uuid),
                            title=doc.title,
                            category=doc.category,
                            year=doc.year,
                            result=result,
                        )
                        async with report_lock:
                            report.processed += 1
                except Exception as e:
                    async with report_lock:
                        report.failed += 1
                        err = f"{doc.title} ({doc.source_id[:12]}…): {e}"
                        report.errors.append(err)
                    logger.exception("Injection failed for {}", doc.title)
                    await tracker.mark_failed(
                        doc.source_id,
                        error=str(e),
                        title=doc.title,
                        category=doc.category,
                    )
                finally:
                    if file_path:
                        Path(file_path).unlink(missing_ok=True)

        await asyncio.gather(*[handle_one(cat, doc) for cat, doc in all_docs])
        return report

    @staticmethod
    def _source_for_category(
        sources: List[tuple[str, BaseDocumentSource]],
        cat_key: str,
    ) -> BaseDocumentSource:
        for k, src in sources:
            if k == cat_key:
                return src
        raise KeyError(cat_key)

    @staticmethod
    def _safe_document_name(doc: DocumentInfo) -> str:
        name = doc.title or "document"
        for ch in '<>:"/\\|?*':
            name = name.replace(ch, "_")
        return name[:200] if name else "document"

    @staticmethod
    def _original_filename(doc: DocumentInfo, file_path: str) -> str:
        if doc.local_path:
            return Path(doc.local_path).name
        if doc.url:
            from urllib.parse import urlparse

            p = urlparse(doc.url).path
            if p and p.lower().endswith(".pdf"):
                return Path(p).name
        return Path(file_path).name

    async def _ingest_file(
        self,
        file_path: str,
        document_name: str,
        original_filename: Optional[str],
    ) -> Dict[str, Any]:
        form = ParseFormData()
        request_config = RequestConfigBuilder.from_form_data(form.model_dump())
        factory = ParserFactory(request_config)
        parsed_response = await factory.parse(file_path)

        document_id = str(uuid.uuid4())
        ingestion_service = IngestionService()
        gathered = ingestion_service.gather_document_data(
            parsed_response=parsed_response,
            document_name=document_name,
        )
        if gathered["status"] != "success":
            raise RuntimeError(gathered.get("error") or "gather_document_data failed")

        pdf_storage_result = None
        if file_path.lower().endswith(".pdf"):
            try:
                pdf_storage_service = PDFStorageService(dpi=300, quality=100)
                pdf_storage_result = await pdf_storage_service.save_pdf_and_images(
                    pdf_path=file_path,
                    document_id=document_id,
                    filename=original_filename,
                    metadata={
                        "document_name": document_name,
                        "file_type": "pdf",
                    },
                )
            except Exception as e:
                logger.warning("PDF storage failed (continuing): {}", e)

        enable_embeddings = bool(getattr(self._settings, "openai_api_key", None))
        enable_vector_db = bool(getattr(self._settings, "qdrant_url", None))

        processing_service = ProcessingService(
            chunk_size=1000,
            chunk_overlap=200,
            respect_sections=True,
            generate_embeddings=enable_embeddings,
            store_in_vector_db=enable_vector_db,
        )
        processing_result = processing_service.process_document(
            raw_ocr=gathered["raw_ocr"],
            page_scalars=gathered["page_scalars"],
            page_images=gathered["page_images"],
            document_name=document_name,
        )
        processing_result["document_id"] = document_id
        if pdf_storage_result:
            processing_result["pdf_storage"] = {
                "pdf_file_id": pdf_storage_result["pdf_file_id"],
                "image_file_ids": pdf_storage_result["image_file_ids"],
                "page_count": pdf_storage_result["page_count"],
            }
        return processing_result
