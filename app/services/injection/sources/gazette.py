"""Nepal Gazette / Rajpatra — crawl index pages for PDF links."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

import httpx
from loguru import logger

from app.services.injection.scrape_utils import extract_pdf_links, fetch_text, polite_delay
from app.services.injection.sources.base import BaseDocumentSource, DocumentInfo, year_in_range


class GazetteSource(BaseDocumentSource):
    async def discover(
        self,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[DocumentInfo]:
        base = (self.config.get("base_url") or "").rstrip("/")
        paths = self.config.get("index_paths") or ["/"]
        items: List[DocumentInfo] = []
        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(headers={"User-Agent": "dafa.ai-injection/1.0"}, timeout=timeout) as client:
            for path in paths:
                page_url = f"{base}{path}" if path.startswith("/") else f"{base}/{path}"
                try:
                    await polite_delay(self.scraper_delay_s)
                    html = await fetch_text(page_url, client)
                    for pdf_url in extract_pdf_links(html, page_url):
                        title = Path(pdf_url).stem.replace("-", " ").replace("_", " ")
                        y = self._year_from_text(title)
                        if not year_in_range(y, year_from, year_to):
                            continue
                        sid = DocumentInfo.make_source_id(self.category_key, pdf_url, None, title)
                        items.append(
                            DocumentInfo(
                                title=title or pdf_url,
                                category=self.category_key,
                                source_id=sid,
                                url=pdf_url,
                                year=y,
                                metadata={"listing_page": page_url, "issuer": "Gazette"},
                            )
                        )
                except Exception as e:
                    logger.warning("Gazette discover failed for {}: {}", page_url, e)
        return items

    @staticmethod
    def _year_from_text(title: str) -> Optional[int]:
        import re

        m = re.search(r"(20\d{2})", title)
        return int(m.group(1)) if m else None

    async def fetch(self, doc_info: DocumentInfo) -> str:
        if not doc_info.url:
            raise ValueError("url required")
        timeout = httpx.Timeout(120.0)
        async with httpx.AsyncClient(headers={"User-Agent": "dafa.ai-injection/1.0"}, timeout=timeout) as client:
            await polite_delay(self.scraper_delay_s)
            resp = await client.get(doc_info.url, follow_redirects=True)
            resp.raise_for_status()
            suffix = Path(doc_info.url).suffix or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resp.content)
                return tmp.name
