"""Save fetched PDFs to disk grouped by injection source (acts_rules, finance_acts, …).

For ``finance_acts``, files are stored as::

    {base}/finance_acts/{kind}/{bs_year}/{title}.pdf

where *kind* is ``annual_finance_act`` (IRD yearly आर्थिक ऐन), ``law_commission`` (GIWMS / Law Commission),
or ``local`` (dropped into data/documents). *bs_year* is the parsed Bikram Sambat year, or ``unknown_year``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from app.services.injection.sources.base import DocumentInfo
from app.settings import Settings


def resolve_injection_download_dir(settings: Settings) -> Path:
    p = Path(settings.injection_download_dir)
    if p.is_absolute():
        return p
    backend_root = Path(__file__).resolve().parents[3]
    return (backend_root / p).resolve()


def sanitize_act_filename(title: str, max_len: int = 160) -> str:
    """Filesystem-safe stem from act title (keeps Nepali/Latin letters, digits)."""
    t = title.strip()
    t = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", t)
    t = re.sub(r"_+", "_", t).strip("._ ")
    if not t:
        t = "document"
    if len(t) > max_len:
        t = t[: max_len - 3].rstrip("_") + "..."
    return t


def _finance_acts_kind_folder(doc: DocumentInfo) -> str:
    """Subfolder under finance_acts: source of the document."""
    md = doc.metadata or {}
    if md.get("annual_finance_act"):
        return "annual_finance_act"
    if md.get("source") == "local":
        return "local"
    return "law_commission"


def _structured_subdir(source_key: str, doc: DocumentInfo) -> Path:
    """Extra path segments under source_key (e.g. finance_acts/annual_finance_act/2082/)."""
    if source_key != "finance_acts":
        return Path()
    kind = _finance_acts_kind_folder(doc)
    year_seg = str(doc.year) if doc.year is not None else "unknown_year"
    return Path(kind) / year_seg


def finance_acts_storage_info(doc: DocumentInfo) -> dict[str, str]:
    """Kind + BS year folder used under ``finance_acts/`` (for logs and API)."""
    return {
        "kind": _finance_acts_kind_folder(doc),
        "year": str(doc.year) if doc.year is not None else "unknown_year",
    }


def act_pdf_destination(base_dir: Path, source_key: str, doc: DocumentInfo) -> Path:
    """
    Unique path: ``{base_dir}/{source_key}/[kind/year/]{title}.pdf`` for finance_acts;
    otherwise ``{base_dir}/{source_key}/{title}.pdf``.
    Appends short hash if a file with the same name already exists.
    """
    folder = base_dir / source_key / _structured_subdir(source_key, doc)
    folder.mkdir(parents=True, exist_ok=True)
    stem = sanitize_act_filename(doc.title)
    name = f"{stem}.pdf"
    dest = folder / name
    if not dest.exists():
        return dest
    short = doc.source_id[:8]
    alt = folder / f"{stem}_{short}.pdf"
    if not alt.exists():
        return alt
    return folder / f"{stem}_{short}_{doc.source_id[8:16]}.pdf"


def copy_fetched_pdf_to_download_dir(
    src_temp_path: str,
    dest: Path,
) -> None:
    """Copy from temp file to final path (overwrite if same path somehow)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_temp_path, dest)
