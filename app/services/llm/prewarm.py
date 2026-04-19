"""Pre-warm Gemini context caches at server startup.

First user after a restart pays ~7s creating the navigator and synthesis
caches. Running this at startup in a background task shifts that cost
off the user-facing path.

Each pre-warm is best-effort: any failure is logged and ignored — the
runtime path creates the cache lazily anyway.
"""

import asyncio
import json
from typing import Iterable

from loguru import logger

from app.prompts.factory import get_prompts
from app.services.llm.service import LLMService
from app.services.retrieval.page_index.cache import CACHE_DIR, get_cached_tree
from app.services.retrieval.page_index.section_retriever import SectionRetriever


# Cap per-restart prewarm cost — each navigator cache creation burns
# ~36K input tokens (~$0.011) and ~3s of wall clock.
_MAX_NAVIGATOR_PREWARMS = 5


def _file_cached_document_ids() -> list[str]:
    """Document IDs that already have a file-cached tree, ordered by recency
    of their tree.json mtime (most recent first) — a decent proxy for "likely
    to be queried soon after restart".
    """
    if not CACHE_DIR.exists():
        return []
    candidates: list[tuple[float, str]] = []
    for p in CACHE_DIR.iterdir():
        if not p.is_dir():
            continue
        tree_path = p / "tree.json"
        if not tree_path.exists():
            continue
        try:
            mtime = tree_path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, p.name))
    candidates.sort(reverse=True)
    return [doc_id for _, doc_id in candidates]


def _warm_synthesis_cache(llm: LLMService, languages: Iterable[str]) -> None:
    """Create the MeroDafa answer-synthesis system-prompt cache for each language."""
    for lang in languages:
        try:
            system_prompt, _ = get_prompts("page_index", "answer_synthesis", lang)
            name = llm._get_or_create_system_cache(system_prompt)
            if name:
                logger.info(f"[prewarm] synthesis cache ready ({lang}): {name}")
        except Exception as e:
            logger.warning(f"[prewarm] synthesis cache failed for {lang}: {e}")


def _warm_navigator_cache(llm: LLMService, document_ids: list[str]) -> None:
    """Create the navigator system+tree cache for each document we have on disk."""
    retriever = SectionRetriever(llm_service=llm)
    for doc_id in document_ids:
        try:
            tree_doc = get_cached_tree(doc_id)
            if not tree_doc:
                continue
            tree = tree_doc.get("tree", {})
            if not tree or not tree.get("nodes"):
                continue
            language = tree_doc.get("language", "en")
            compact = retriever._build_compact_tree(tree)
            tree_json = json.dumps(compact, ensure_ascii=False, indent=2)
            system_prompt, _ = get_prompts("page_index", "tree_navigator", language)
            system_with_tree = f"{system_prompt}\n\n---\nDocument Tree:\n{tree_json}"
            name = llm._get_or_create_system_cache(system_with_tree)
            if name:
                logger.info(f"[prewarm] navigator cache ready for {doc_id[:8]}: {name}")
        except Exception as e:
            logger.warning(f"[prewarm] navigator cache failed for {doc_id[:8]}: {e}")


async def prewarm_caches() -> None:
    """Entry point — runs the warm-up off the event loop so startup stays snappy."""
    def _run() -> None:
        llm = LLMService()
        if not llm.client:
            logger.info("[prewarm] no Gemini client; skipping")
            return
        doc_ids = _file_cached_document_ids()[:_MAX_NAVIGATOR_PREWARMS]
        logger.info(
            f"[prewarm] starting (synthesis langs=en,ne; navigator docs={len(doc_ids)}/"
            f"{_MAX_NAVIGATOR_PREWARMS})"
        )
        _warm_synthesis_cache(llm, ["en", "ne"])
        _warm_navigator_cache(llm, doc_ids)
        logger.info("[prewarm] done")

    await asyncio.to_thread(_run)
