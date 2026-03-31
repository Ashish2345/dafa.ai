"""Local directory document source."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.services.injection.scrape_utils import guess_bs_year
from app.services.injection.sources.base import BaseDocumentSource, DocumentInfo, year_in_range


def _backend_root() -> Path:
    """backend/app/services/injection/sources/local.py -> backend."""
    return Path(__file__).resolve().parents[4]


class LocalDirectorySource(BaseDocumentSource):
    """Discover PDFs under configured directories."""

    async def discover(
        self,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[DocumentInfo]:
        items: List[DocumentInfo] = []
        for block in self.config.get("directories") or []:
            rel = block.get("path", "")
            cat = block.get("category") or self.category_key
            root = Path(rel).expanduser()
            if not root.is_absolute():
                # Resolve relative to backend root (not process CWD) so inject works from any directory
                root = _backend_root() / root
            root = root.resolve()
            if not root.exists():
                logger.warning(
                    "Local injection: directory does not exist (skipping): {} — create it or put PDFs under backend/data/documents",
                    root,
                )
                continue
            pdfs = sorted(p for p in root.rglob("*.pdf") if p.is_file())
            if not pdfs:
                logger.warning(
                    "Local injection: no PDF files under {} (add .pdf files or update config/sources.yaml)",
                    root,
                )
            for path in pdfs:
                title = path.stem.replace("_", " ")
                year = guess_bs_year(path.name)
                if not year_in_range(year, year_from, year_to):
                    continue
                sid = DocumentInfo.make_source_id(cat, None, str(path.resolve()), title)
                items.append(
                    DocumentInfo(
                        title=title,
                        category=cat,
                        source_id=sid,
                        local_path=str(path.resolve()),
                        year=year,
                        metadata={"filename": path.name, "source": "local"},
                    )
                )
        return items

    async def fetch(self, doc_info: DocumentInfo) -> str:
        if not doc_info.local_path:
            raise ValueError("local_path required for LocalDirectorySource.fetch")
        src = Path(doc_info.local_path)
        if not src.exists():
            raise FileNotFoundError(src)
        suffix = src.suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            dst = Path(tmp.name)
        shutil.copy2(src, dst)
        return str(dst)
