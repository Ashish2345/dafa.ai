"""HTTP helpers for injection sources (polite delays, PDF link extraction)."""

from __future__ import annotations

import asyncio
import re
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

_GIWMS_PDF_RE = re.compile(
    r"https://giwmscdntwo\.gov\.np/media/pdf_upload/[^\s\"'<>]+\.pdf",
    re.IGNORECASE,
)
# IRD (and similar) article pages embed flipbook PDFs: var pdf = 'https://giwmscdntwo.gov.np/media/pdf/….pdf';
_GIWMS_MEDIA_PDF_RE = re.compile(
    r"https://giwmscdntwo\.gov\.np/media/pdf/[^\s\"'<>]+\.pdf",
    re.IGNORECASE,
)
_VAR_PDF_FLIPBOOK_RE = re.compile(
    r"var\s+pdf\s*=\s*['\"](https://giwmscdntwo\.gov\.np/media/pdf/[^'\"]+\.pdf)['\"]",
    re.IGNORECASE,
)


def extract_all_giwms_pdf_urls(html: str) -> List[str]:
    """GIWMS-hosted PDFs: bare URLs in HTML plus `var pdf = '…'` flipbook embeds (IRD detail pages)."""
    seen: Set[str] = set()
    out: List[str] = []
    for rx in (_GIWMS_PDF_RE, _GIWMS_MEDIA_PDF_RE):
        for m in rx.finditer(html):
            u = m.group(0)
            if u not in seen:
                seen.add(u)
                out.append(u)
    for m in _VAR_PDF_FLIPBOOK_RE.finditer(html):
        u = m.group(1)
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def parse_ird_finance_category_cards(html: str) -> List[tuple[str, str]]:
    """IRD category listing (e.g. /category/financeact/): (content path, title) from each card."""
    soup = BeautifulSoup(html, "html.parser")
    grid = soup.select_one("div.category-1-grid")
    if not grid:
        return []
    out: List[tuple[str, str]] = []
    for card in grid.select(":scope > div.grid__card"):
        title_a = card.select_one("h3.card__title a[href]")
        if not title_a:
            continue
        href = re.sub(r"\s+", "", (title_a.get("href") or "").strip())
        title = title_a.get_text(strip=True)
        if href.startswith("/content/"):
            out.append((href, title))
    return out

_NEPALI_DIGIT_TRANS = str.maketrans("०१२३४५६७८९", "0123456789")


def absolute_url(base: str, href: str) -> str:
    return urljoin(base.rstrip("/") + "/", href.strip())


def normalize_nepali_digits(text: str) -> str:
    """Devanagari numerals (often used for BS years in filenames) → Western digits."""
    return text.translate(_NEPALI_DIGIT_TRANS)


def guess_bs_year(text: str) -> Optional[int]:
    """Parse a 4-digit BS year (20xx) from text after normalizing Nepali digits."""
    t = normalize_nepali_digits(text)
    m = re.search(r"(20\d{2})", t)
    return int(m.group(1)) if m else None


def extract_giwms_pdf_urls(html: str) -> List[str]:
    """Direct PDF links on Nepal Law Commission (GIWMS CDN) pages."""
    seen: Set[str] = set()
    out: List[str] = []
    for m in _GIWMS_PDF_RE.finditer(html):
        url = m.group(0)
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def extract_volume_list_category_urls(html: str, list_page_url: str) -> List[str]:
    """Category URLs from the 'ऐन संग्रहको सूची' / list-volume-act table (in-cell-link only)."""
    soup = BeautifulSoup(html, "html.parser")
    seen: Set[str] = set()
    out: List[str] = []
    for a in soup.select("a.in-cell-link[href*='/category/']"):
        href = (a.get("href") or "").strip()
        if not href or not re.search(r"/category/\d+", href):
            continue
        full = absolute_url(list_page_url, href)
        if full not in seen:
            seen.add(full)
            out.append(full)
    return sorted(out)


def extract_pdf_links(html: str, page_url: str) -> List[str]:
    """Return absolute URLs for .pdf links found in HTML."""
    soup = BeautifulSoup(html, "html.parser")
    seen: Set[str] = set()
    out: List[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("#", "javascript:")):
            continue
        full = absolute_url(page_url, href)
        path = urlparse(full).path.lower()
        if path.endswith(".pdf"):
            if full not in seen:
                seen.add(full)
                out.append(full)
    return out


def lawcommission_browser_headers() -> dict[str, str]:
    """Headers that reduce 403s from lawcommission.gov.np / GIWMS."""
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ne-NP,ne;q=0.9,en-US;q=0.8,en;q=0.7",
    }


async def fetch_text(
    url: str,
    client: httpx.AsyncClient,
    timeout: float = 60.0,
    *,
    referer: Optional[str] = None,
    retries: int = 3,
) -> str:
    """GET HTML with optional Referer and retries on 403 (WAF / rate limit)."""
    extra: dict[str, str] = {}
    if referer:
        extra["Referer"] = referer
    for attempt in range(retries + 1):
        resp = await client.get(url, timeout=timeout, follow_redirects=True, headers=extra)
        if resp.status_code == 403 and attempt < retries:
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.text


async def polite_delay(seconds: float) -> None:
    if seconds > 0:
        await asyncio.sleep(seconds)
