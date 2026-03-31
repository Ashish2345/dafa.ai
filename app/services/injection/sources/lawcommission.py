"""Nepal Law Commission — GIWMS volume-list crawl (list-volume-act → category pages → PDFs)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

import httpx
from loguru import logger

from app.services.injection.scrape_utils import lawcommission_browser_headers, polite_delay
from app.services.injection.sources.base import BaseDocumentSource, DocumentInfo
from app.services.injection.sources.giwms_crawl import discover_from_giwms_volume_list


class LawCommissionSource(BaseDocumentSource):
    async def discover(
        self,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[DocumentInfo]:
        base = (self.config.get("base_url") or "https://lawcommission.gov.np").rstrip("/")
        path = self.config.get("volume_list_path") or "/pages/list-volume-act/"
        max_cat = self.config.get("max_categories")
        max_categories = int(max_cat) if max_cat is not None else None
        raw_ids = self.config.get("category_ids")
        category_id_allowlist: Optional[List[int]] = None
        if raw_ids:
            category_id_allowlist = [int(x) for x in raw_ids]

        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(
            headers=lawcommission_browser_headers(),
            timeout=timeout,
        ) as client:
            try:
                return await discover_from_giwms_volume_list(
                    base_url=base,
                    volume_list_path=path,
                    result_category_key=self.category_key,
                    scraper_delay_s=self.scraper_delay_s,
                    client=client,
                    year_from=year_from,
                    year_to=year_to,
                    filename_keywords=None,
                    max_categories=max_categories,
                    category_id_allowlist=category_id_allowlist,
                )
            except Exception as e:
                logger.warning("Law Commission GIWMS discover failed: {}", e)
                return []

    async def fetch(self, doc_info: DocumentInfo) -> str:
        if not doc_info.url:
            raise ValueError("url required")
        timeout = httpx.Timeout(120.0)
        async with httpx.AsyncClient(
            headers=lawcommission_browser_headers(),
            timeout=timeout,
        ) as client:
            await polite_delay(self.scraper_delay_s)
            resp = await client.get(doc_info.url, follow_redirects=True)
            resp.raise_for_status()
            suffix = Path(doc_info.url).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resp.content)
                return tmp.name
