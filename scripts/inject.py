"""
Batch document injection CLI.

Run from the backend directory:
  uv run python -m scripts.inject --help
  uv run python -m scripts.inject --year-from 2080 --dry-run
  uv run python -m scripts.inject --category finance_acts --year-from 2081 --year-to 2081
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    backend_root = Path(__file__).resolve().parent.parent
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


_bootstrap_path()

try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)
except ImportError:
    pass


from loguru import logger  # noqa: E402

from app.db.mongodb import mongodb  # noqa: E402
from app.services.injection.local_save import resolve_injection_download_dir  # noqa: E402
from app.services.injection.orchestrator import BatchInjectionOrchestrator  # noqa: E402
from app.services.injection.tracker import (  # noqa: E402
    InjectionTracker,
    LocalInjectionTracker,
    resolve_local_injection_log_path,
)
from app.settings import settings  # noqa: E402


def _parse_categories(raw: list[str] | None) -> list[str] | None:
    if not raw:
        return None
    out: list[str] = []
    for part in raw:
        for item in part.split(","):
            s = item.strip()
            if s:
                out.append(s)
    return out or None


async def _run() -> int:
    parser = argparse.ArgumentParser(description="Inject documents from configured sources into RAG storage.")
    parser.add_argument(
        "--year-from",
        type=int,
        default=None,
        help="Inclusive BS year filter (e.g. 2080)",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        default=None,
        help="Inclusive BS year filter",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=None,
        help="Source key from config/sources.yaml (repeatable). Example: --category finance_acts",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover only; do not fetch or ingest")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if source_id already succeeded")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override INJECTION_CONCURRENCY for this run",
    )
    parser.add_argument("--stats", action="store_true", help="Print injection_log summary and exit")
    parser.add_argument(
        "--stats-category",
        type=str,
        default=None,
        help="Filter --stats by injection category key",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Fetch PDFs and save under <dir>/<source_key>/<act title>.pdf (no OCR/embed/Qdrant)",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Base folder for --download-only (default: INJECTION_DOWNLOAD_DIR / data/downloaded)",
    )
    args = parser.parse_args()

    if args.download_only and args.dry_run:
        parser.error("--download-only cannot be used with --dry-run")

    mongo_connected = False
    local_tracker: LocalInjectionTracker | None = None

    try:
        await mongodb.connect(settings)
        mongo_connected = True
    except Exception as exc:
        log_path = resolve_local_injection_log_path(settings)
        logger.warning(
            "MongoDB connection failed ({}); using local injection log at {}",
            exc,
            log_path,
        )
        local_tracker = LocalInjectionTracker(log_path)

    try:
        if args.stats:
            tracker = (
                local_tracker
                if local_tracker is not None
                else InjectionTracker(mongodb.get_database())
            )
            summary = await tracker.get_stats(category=args.stats_category)
            print(json.dumps(summary, indent=2))
            return 0

        download_only_path: Path | None = None
        if args.download_only:
            if args.download_dir:
                p = Path(args.download_dir)
                backend_root = Path(__file__).resolve().parent.parent
                download_only_path = p.resolve() if p.is_absolute() else (backend_root / p).resolve()
            else:
                download_only_path = resolve_injection_download_dir(settings)

        orch = BatchInjectionOrchestrator(settings)
        categories = _parse_categories(args.categories)
        report = await orch.run(
            categories=categories,
            year_from=args.year_from,
            year_to=args.year_to,
            force=args.force,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
            tracker=local_tracker,
            download_only_dir=download_only_path,
        )
        if args.dry_run:
            print("\n--- dry-run: discovered documents ---", flush=True)
            if report.discovered_items:
                for it in report.discovered_items:
                    loc = it.get("url") or it.get("local_path") or ""
                    print(
                        f"  [{it['source_key']}] {it.get('title')} | year={it.get('year')} | {loc}",
                        flush=True,
                    )
            else:
                print(
                    "  (none — check network/403, config category_ids, filename_keywords, or year filters)",
                    flush=True,
                )
            print("--- end ---\n", flush=True)
        if args.download_only and report.downloaded_items:
            print("\n--- saved PDFs (by source / act) ---", flush=True)
            for it in report.downloaded_items:
                print(f"  [{it['source_key']}] {it.get('path')}", flush=True)
            print(f"--- base: {download_only_path} ---\n", flush=True)
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.failed == 0 else 1
    finally:
        if mongo_connected:
            await mongodb.disconnect()


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
