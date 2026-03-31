"""Track batch injection state in MongoDB (`injection_log`) or a local JSON file."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.settings import Settings


def resolve_local_injection_log_path(settings: Settings) -> Path:
    """Resolve `injection_local_log_path` relative to backend root."""
    p = Path(settings.injection_local_log_path)
    if p.is_absolute():
        return p
    backend_root = Path(__file__).resolve().parents[3]
    return (backend_root / p).resolve()


class InjectionTracker:
    """Deduplication and run history for injected documents."""

    COLLECTION = "injection_log"

    def __init__(self, database) -> None:
        self._db = database

    def _col(self):
        return self._db[self.COLLECTION]

    async def is_ingested(self, source_id: str) -> bool:
        doc = await self._col().find_one({"source_id": source_id, "status": "success"})
        return doc is not None

    async def mark_in_progress(self, source_id: str, title: str, category: str, year: Optional[int]) -> None:
        now = datetime.now(timezone.utc)
        await self._col().update_one(
            {"source_id": source_id},
            {
                "$set": {
                    "source_id": source_id,
                    "title": title,
                    "category": category,
                    "year": year,
                    "status": "in_progress",
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def mark_ingested(
        self,
        source_id: str,
        document_id: str,
        title: str,
        category: str,
        year: Optional[int],
        result: Dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc)
        collection_name = result.get("collection_name") or result.get("qdrant_collection")
        meta = result.get("metadata") or {}
        if not collection_name and isinstance(meta, dict):
            collection_name = meta.get("act_name")  # may be act label, not qdrant name
        update = {
            "source_id": source_id,
            "document_id": document_id,
            "title": title,
            "category": category,
            "year": year,
            "status": "success",
            "result_summary": {
                "vector_store_status": result.get("vector_store_status"),
                "chunks": len(result.get("chunks") or []),
                "collection_name": collection_name,
            },
            "updated_at": now,
            "ingested_at": now,
            "error": None,
        }
        await self._col().update_one(
            {"source_id": source_id},
            {"$set": update, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    async def mark_failed(self, source_id: str, error: str, title: str = "", category: str = "") -> None:
        now = datetime.now(timezone.utc)
        await self._col().update_one(
            {"source_id": source_id},
            {
                "$set": {
                    "source_id": source_id,
                    "title": title,
                    "category": category,
                    "status": "failed",
                    "error": error[:8000],
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def get_stats(self, category: Optional[str] = None) -> Dict[str, Any]:
        match: Dict[str, Any] = {}
        if category:
            match["category"] = category
        col = self._col()
        pipeline: List[Dict[str, Any]] = []
        if match:
            pipeline.append({"$match": match})
        pipeline.append({"$group": {"_id": "$status", "count": {"$sum": 1}}})
        cursor = col.aggregate(pipeline)
        by_status: Dict[str, int] = {}
        async for row in cursor:
            sid = row.get("_id") or "unknown"
            by_status[str(sid)] = int(row.get("count", 0))
        total = await col.count_documents(match if match else {})
        return {"by_status": by_status, "total_documents": total}

    async def delete_by_source_id(self, source_id: str) -> int:
        res = await self._col().delete_many({"source_id": source_id})
        logger.info(f"Deleted {res.deleted_count} injection_log row(s) for source_id={source_id}")
        return res.deleted_count

    async def reset(self, source_id: Optional[str] = None) -> None:
        """
        Remove tracking record(s). If source_id is set, delete that row only.
        If None, logs a warning and does nothing (avoid accidental full wipe).
        """
        if source_id:
            await self.delete_by_source_id(source_id)
        else:
            logger.warning("InjectionTracker.reset(None) is a no-op; pass a specific source_id to delete.")


def _dt_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class LocalInjectionTracker:
    """
    File-backed injection log when MongoDB is unavailable.
    Same async API as InjectionTracker; stores JSON at `path`.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    def _load(self) -> Dict[str, Any]:
        if not self._path.exists():
            return {"records": {}}
        with self._path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "records" not in data:
            return {"records": {}}
        if not isinstance(data["records"], dict):
            data["records"] = {}
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
        tmp.replace(self._path)

    async def is_ingested(self, source_id: str) -> bool:
        async with self._lock:
            rec = self._load()["records"].get(source_id)
            return rec is not None and rec.get("status") == "success"

    async def mark_in_progress(self, source_id: str, title: str, category: str, year: Optional[int]) -> None:
        async with self._lock:
            data = self._load()
            now = _dt_now_iso()
            prev = data["records"].get(source_id, {})
            created = prev.get("created_at", now)
            data["records"][source_id] = {
                **prev,
                "source_id": source_id,
                "title": title,
                "category": category,
                "year": year,
                "status": "in_progress",
                "updated_at": now,
                "created_at": created,
            }
            self._save(data)

    async def mark_ingested(
        self,
        source_id: str,
        document_id: str,
        title: str,
        category: str,
        year: Optional[int],
        result: Dict[str, Any],
    ) -> None:
        async with self._lock:
            data = self._load()
            now = _dt_now_iso()
            collection_name = result.get("collection_name") or result.get("qdrant_collection")
            meta = result.get("metadata") or {}
            if not collection_name and isinstance(meta, dict):
                collection_name = meta.get("act_name")
            prev = data["records"].get(source_id, {})
            created = prev.get("created_at", now)
            data["records"][source_id] = {
                "source_id": source_id,
                "document_id": document_id,
                "title": title,
                "category": category,
                "year": year,
                "status": "success",
                "result_summary": {
                    "vector_store_status": result.get("vector_store_status"),
                    "chunks": len(result.get("chunks") or []),
                    "collection_name": collection_name,
                },
                "updated_at": now,
                "ingested_at": now,
                "error": None,
                "created_at": created,
            }
            self._save(data)

    async def mark_failed(self, source_id: str, error: str, title: str = "", category: str = "") -> None:
        async with self._lock:
            data = self._load()
            now = _dt_now_iso()
            prev = data["records"].get(source_id, {})
            created = prev.get("created_at", now)
            data["records"][source_id] = {
                **prev,
                "source_id": source_id,
                "title": title,
                "category": category,
                "status": "failed",
                "error": error[:8000],
                "updated_at": now,
                "created_at": created,
            }
            self._save(data)

    async def get_stats(self, category: Optional[str] = None) -> Dict[str, Any]:
        async with self._lock:
            data = self._load()
            records = data["records"].values()
            by_status: Dict[str, int] = {}
            total = 0
            for rec in records:
                if category and rec.get("category") != category:
                    continue
                total += 1
                st = str(rec.get("status") or "unknown")
                by_status[st] = by_status.get(st, 0) + 1
            return {"by_status": by_status, "total_documents": total, "storage": "local", "path": str(self._path)}

    async def delete_by_source_id(self, source_id: str) -> int:
        async with self._lock:
            data = self._load()
            if source_id in data["records"]:
                del data["records"][source_id]
                self._save(data)
                logger.info("Deleted local injection_log row for source_id={}", source_id)
                return 1
            return 0

    async def reset(self, source_id: Optional[str] = None) -> None:
        if source_id:
            await self.delete_by_source_id(source_id)
        else:
            logger.warning("LocalInjectionTracker.reset(None) is a no-op; pass a specific source_id to delete.")
