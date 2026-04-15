"""
File-based cache for PageIndex trees and markdown.

Stores tree JSON and markdown text on disk so repeated queries skip MongoDB.
Cache is invalidated when a document is re-ingested (save_tree writes new data).

Cache directory: .cache/page_index/<document_id>/
  tree.json    — full tree document (tree, language, image_dimensions)
  markdown.txt — full markdown text
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger

CACHE_DIR = Path(__file__).resolve().parents[4] / ".cache" / "page_index"


def _doc_dir(document_id: str) -> Path:
    return CACHE_DIR / document_id


def get_cached_tree(document_id: str) -> Optional[dict]:
    """Return cached tree document or None."""
    path = _doc_dir(document_id) / "tree.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.debug(f"Cache HIT: tree for {document_id[:8]}")
        return data
    except Exception as e:
        logger.warning(f"Cache read failed for tree {document_id[:8]}: {e}")
        return None


def get_cached_markdown(document_id: str) -> Optional[str]:
    """Return cached markdown or None."""
    path = _doc_dir(document_id) / "markdown.txt"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        logger.debug(f"Cache HIT: markdown for {document_id[:8]}")
        return text
    except Exception as e:
        logger.warning(f"Cache read failed for markdown {document_id[:8]}: {e}")
        return None


def save_to_cache(document_id: str, tree_doc: dict, markdown: str) -> None:
    """Write tree and markdown to disk cache."""
    try:
        doc_dir = _doc_dir(document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Strip _id and convert datetimes (not JSON serializable)
        tree_data = {k: v for k, v in tree_doc.items() if k != "_id"}
        (doc_dir / "tree.json").write_text(
            json.dumps(tree_data, ensure_ascii=False, default=str), encoding="utf-8",
        )
        (doc_dir / "markdown.txt").write_text(markdown, encoding="utf-8")
        logger.info(f"Cached tree + markdown for {document_id[:8]}")
    except Exception as e:
        logger.warning(f"Cache write failed for {document_id[:8]}: {e}")


def invalidate_cache(document_id: str) -> None:
    """Remove cached data for a document (call on re-ingestion)."""
    doc_dir = _doc_dir(document_id)
    if doc_dir.exists():
        import shutil
        shutil.rmtree(doc_dir, ignore_errors=True)
        logger.info(f"Cache invalidated for {document_id[:8]}")
