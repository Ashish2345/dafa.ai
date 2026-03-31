"""Finance / revenue legislation on Nepal Law Commission (GIWMS). Defaults: खण्ड ६, all PDFs unless filename_keywords is set.

Optional: annual आर्थिक ऐन (Finance Act) PDFs from IRD https://ird.gov.np/category/financeact/ — see ird_finance_* in sources.yaml.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Set

import httpx
from loguru import logger

from app.services.injection.scrape_utils import (
    extract_all_giwms_pdf_urls,
    fetch_text,
    guess_bs_year,
    lawcommission_browser_headers,
    parse_ird_finance_category_cards,
    polite_delay,
)
from app.services.injection.sources.base import BaseDocumentSource, DocumentInfo, year_in_range
from app.services.injection.sources.giwms_crawl import discover_from_giwms_volume_list


async def _discover_ird_annual_finance_acts(
    *,
    client: httpx.AsyncClient,
    ird_base: str,
    category_paths: List[str],
    max_pages: int,
    scraper_delay_s: float,
    result_category_key: str,
    year_from: Optional[int],
    year_to: Optional[int],
) -> List[DocumentInfo]:
    """Paginated IRD category pages → article HTML → embedded GIWMS PDF URLs."""
    items: List[DocumentInfo] = []
    seen_pdf: Set[str] = set()
    seen_content: Set[str] = set()
    ird_base = ird_base.rstrip("/")

    for raw_path in category_paths:
        cat_path = raw_path.strip()
        if not cat_path.startswith("/"):
            cat_path = f"/{cat_path}"
        if not cat_path.endswith("/"):
            cat_path = f"{cat_path}/"

        for page in range(1, max(1, max_pages) + 1):
            listing_url = f"{ird_base}{cat_path}"
            if page > 1:
                listing_url = f"{listing_url}?page={page}"
            try:
                await polite_delay(scraper_delay_s)
                html = await fetch_text(listing_url, client, referer=f"{ird_base}/")
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 404:
                    break
                logger.warning("IRD finance listing failed {}: {}", listing_url, e)
                break
            except Exception as e:
                logger.warning("IRD finance listing failed {}: {}", listing_url, e)
                break

            cards = parse_ird_finance_category_cards(html)
            if not cards:
                break

            for rel_href, title in cards:
                if rel_href in seen_content:
                    continue
                seen_content.add(rel_href)
                article_url = f"{ird_base}{rel_href}"
                y = guess_bs_year(title)
                if not year_in_range(y, year_from, year_to):
                    continue
                try:
                    await polite_delay(scraper_delay_s)
                    art_html = await fetch_text(article_url, client, referer=listing_url)
                except Exception as e:
                    logger.warning("IRD finance act page failed {}: {}", article_url, e)
                    continue
                for pdf_url in extract_all_giwms_pdf_urls(art_html):
                    if pdf_url in seen_pdf:
                        continue
                    seen_pdf.add(pdf_url)
                    sid = DocumentInfo.make_source_id(result_category_key, pdf_url, None, title)
                    items.append(
                        DocumentInfo(
                            title=title,
                            category=result_category_key,
                            source_id=sid,
                            url=pdf_url,
                            year=y,
                            metadata={
                                "issuer": "IRD",
                                "annual_finance_act": True,
                                "article_url": article_url,
                                "listing_page": listing_url,
                            },
                        )
                    )
    return items


class FinanceActsSource(BaseDocumentSource):
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

        # null / [] / omit => no filename filter (all PDFs in selected categories)
        raw_kw = self.config.get("filename_keywords")
        if raw_kw is None or raw_kw == []:
            keywords = None
        else:
            keywords = [str(x) for x in raw_kw]

        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(
            headers=lawcommission_browser_headers(),
            timeout=timeout,
        ) as client:
            items: List[DocumentInfo] = []
            try:
                items = await discover_from_giwms_volume_list(
                    base_url=base,
                    volume_list_path=path,
                    result_category_key=self.category_key,
                    scraper_delay_s=self.scraper_delay_s,
                    client=client,
                    year_from=year_from,
                    year_to=year_to,
                    filename_keywords=keywords,
                    max_categories=max_categories,
                    category_id_allowlist=category_id_allowlist,
                )
            except Exception as e:
                logger.warning("Finance acts GIWMS discover failed: {}", e)

            raw_ird_paths = self.config.get("ird_finance_category_paths")
            if raw_ird_paths:
                try:
                    ird_base = (self.config.get("ird_finance_base_url") or "https://ird.gov.np").rstrip("/")
                    max_pages = int(self.config.get("ird_finance_max_pages") or 20)
                    ird_items = await _discover_ird_annual_finance_acts(
                        client=client,
                        ird_base=ird_base,
                        category_paths=[str(p) for p in raw_ird_paths],
                        max_pages=max_pages,
                        scraper_delay_s=self.scraper_delay_s,
                        result_category_key=self.category_key,
                        year_from=year_from,
                        year_to=year_to,
                    )
                    items.extend(ird_items)
                except Exception as e:
                    logger.warning("IRD annual finance act discover failed: {}", e)

            return items

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
