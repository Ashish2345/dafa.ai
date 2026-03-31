"""Crawl Nepal Law Commission GIWMS site: volume list → category pages → CDN PDFs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote

import httpx
from loguru import logger

from app.services.injection.scrape_utils import (
    extract_giwms_pdf_urls,
    extract_pdf_links,
    extract_volume_list_category_urls,
    fetch_text,
    guess_bs_year,
    polite_delay,
)
from app.services.injection.sources.base import DocumentInfo, year_in_range


def _category_id_from_url(url: str) -> Optional[int]:
    m = re.search(r"/category/(\d+)", url)
    return int(m.group(1)) if m else None


def _filename_matches_keywords(decoded_name: str, keywords: Optional[List[str]]) -> bool:
    if not keywords:
        return True
    normalized = decoded_name.lower()
    for kw in keywords:
        if not kw:
            continue
        if kw in decoded_name:
            return True
        if kw.lower() in normalized:
            return True
    return False


async def discover_from_giwms_volume_list(
    *,
    base_url: str,
    volume_list_path: str,
    result_category_key: str,
    scraper_delay_s: float,
    client: httpx.AsyncClient,
    year_from: Optional[int],
    year_to: Optional[int],
    filename_keywords: Optional[List[str]] = None,
    max_categories: Optional[int] = None,
    category_id_allowlist: Optional[List[int]] = None,
) -> List[DocumentInfo]:
    """
    Fetch /pages/list-volume-act/, follow खण्ड category links, collect giwmscdntwo PDF URLs.
    """
    base = base_url.rstrip("/")
    path = (volume_list_path or "/pages/list-volume-act/").strip()
    if not path.startswith("/"):
        path = "/" + path
    list_url = f"{base}{path}"
    if not list_url.endswith("/"):
        list_url += "/"

    items: List[DocumentInfo] = []
    seen_pdf: set[str] = set()

    await polite_delay(scraper_delay_s)
    list_html = await fetch_text(list_url, client, referer=f"{base}/")
    category_urls = extract_volume_list_category_urls(list_html, list_url)
    if category_id_allowlist:
        allow = set(category_id_allowlist)
        category_urls = [u for u in category_urls if _category_id_from_url(u) in allow]
    if max_categories is not None:
        category_urls = category_urls[: max(0, max_categories)]

    if not category_urls:
        logger.warning("GIWMS volume list produced no category URLs: {}", list_url)

    for cat_url in category_urls:
        cid = _category_id_from_url(cat_url)
        try:
            await polite_delay(scraper_delay_s)
            html = await fetch_text(cat_url, client, referer=list_url)
            pdf_urls = list(dict.fromkeys(extract_giwms_pdf_urls(html) + extract_pdf_links(html, cat_url)))
            for pdf_url in pdf_urls:
                if pdf_url in seen_pdf:
                    continue
                seen_pdf.add(pdf_url)
                raw_name = unquote(Path(pdf_url).name)
                title = raw_name.replace("_", " ").rsplit(".", 1)[0]
                if not _filename_matches_keywords(raw_name, filename_keywords):
                    continue
                y = guess_bs_year(raw_name)
                if not year_in_range(y, year_from, year_to):
                    continue
                sid = DocumentInfo.make_source_id(result_category_key, pdf_url, None, title)
                items.append(
                    DocumentInfo(
                        title=title,
                        category=result_category_key,
                        source_id=sid,
                        url=pdf_url,
                        year=y,
                        metadata={
                            "listing_page": cat_url,
                            "giwms_category_id": cid,
                            "source": "lawcommission_giwms",
                        },
                    )
                )
        except Exception as e:
            logger.warning("GIWMS category crawl failed for {}: {}", cat_url, e)

    return items
