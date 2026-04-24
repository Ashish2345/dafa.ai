"""
Document repository — consolidated document CRUD.
"""

import re
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Kebab-case, ASCII-only, max 80 chars. Strips accents; drops empties."""
    if not text:
        return ""
    lowered = text.strip().lower()
    # Collapse everything not [a-z0-9] into a single hyphen.
    s = _SLUG_RE.sub("-", lowered).strip("-")
    return s[:80]


class DocumentRepository:
    """CRUD for the documents collection."""

    def __init__(self, database):
        self.collection = database.documents

    # ------------------------------------------------------------------
    # Slug allocation
    # ------------------------------------------------------------------

    async def generate_unique_slug(
        self,
        title: str,
        *,
        scope: str = "public",
        user_id: str | None = None,
    ) -> str:
        """Build a kebab-case slug and suffix `-2`, `-3` on collision.

        Uniqueness is scoped:
          - public docs: unique globally among public docs.
          - private docs: unique within the owner's private bucket.
        """
        base = slugify(title) or "document"
        candidate = base
        n = 1
        while True:
            query: dict[str, Any] = {"slug": candidate}
            if scope == "private":
                query["scope"] = "private"
                query["user_id"] = user_id
            else:
                query["$or"] = [
                    {"scope": "public"},
                    {"scope": {"$exists": False}},
                ]
            existing = await self.collection.find_one(query, {"_id": 1})
            if not existing:
                return candidate
            n += 1
            candidate = f"{base}-{n}"

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def save_initial(
        self,
        document_id: str,
        filename: str,
        category: str | None = None,
        title: str | None = None,
        user_id: str | None = None,
        scope: str = "public",
        domain_slug: str | None = None,
        slug: str | None = None,
        icon: str | None = None,
    ) -> None:
        """Create a stub record immediately so callers can poll status.

        Phase 14: ``user_id`` and ``scope`` make documents filterable as either
        a public catalog item (``scope='public'``, ``user_id=None``) or a
        private per-user upload (``scope='private'``, ``user_id=<owner>``).

        Phase 18 (this change): auto-generates a unique ``slug`` from the
        title, accepts ``domain_slug`` to attach the document to a legal
        domain, and accepts ``icon`` (a Lucide icon name from the
        frontend catalog) to override the default document glyph.
        """
        now = datetime.now(timezone.utc)
        effective_title = title or filename
        if not slug:
            slug = await self.generate_unique_slug(
                effective_title, scope=scope, user_id=user_id
            )
        patch: dict[str, Any] = {
            "document_id": document_id,
            "filename": filename,
            "title": effective_title,
            "slug": slug,
            "status": "processing",
            "progress_step": "Queued",
            "updated_at": now,
            "scope": scope,
            "user_id": user_id,
        }
        if category:
            patch["category"] = category
        if domain_slug:
            patch["domain_slug"] = domain_slug
        if icon:
            patch["icon"] = icon
        await self.collection.update_one(
            {"document_id": document_id},
            {"$set": patch, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

    async def update_status(
        self,
        document_id: str,
        status: str,
        step: str = "",
        error: str = "",
    ) -> None:
        """Update processing status and current step in-place."""
        patch: dict[str, Any] = {"status": status, "updated_at": datetime.now(timezone.utc)}
        if step:
            patch["progress_step"] = step
        if error:
            patch["error"] = error
        await self.collection.update_one({"document_id": document_id}, {"$set": patch})

    async def save(
        self,
        document_id: str,
        filename: str,
        metadata: dict[str, Any],
        strategy: str,
        custom_tree_provided: bool = False,
        ingest_warnings: dict | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        await self.collection.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "document_id": document_id,
                    "filename": filename,
                    "metadata": metadata,
                    "strategy": strategy,
                    "status": "completed",
                    "progress_step": "Done",
                    "custom_tree_provided": custom_tree_provided,
                    "ingest_warnings": ingest_warnings,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )

    async def get(self, document_id: str) -> Optional[dict]:
        return await self.collection.find_one({"document_id": document_id}, {"_id": 0})

    async def get_by_slug(
        self,
        slug: str,
        *,
        user_id: str | None = None,
    ) -> Optional[dict]:
        """Resolve a slug to a visible document.

        Visibility rules match ``list_all``: public + legacy docs always match;
        private docs only match when ``user_id`` is the owner.
        """
        query: dict[str, Any] = {
            "slug": slug,
            "$or": [
                {"scope": "public"},
                {"scope": {"$exists": False}},
            ],
        }
        if user_id is not None:
            query["$or"].append({"scope": "private", "user_id": user_id})
        return await self.collection.find_one(query, {"_id": 0})

    async def set_domain(self, document_id: str, domain_slug: str | None) -> bool:
        """Attach / detach a document to a legal domain. ``None`` clears it."""
        patch = (
            {"$unset": {"domain_slug": ""}}
            if domain_slug is None
            else {"$set": {"domain_slug": domain_slug, "updated_at": datetime.now(timezone.utc)}}
        )
        result = await self.collection.update_one({"document_id": document_id}, patch)
        return result.modified_count > 0

    async def set_icon(self, document_id: str, icon: str | None) -> bool:
        """Set / clear the document's Lucide icon name. ``None`` clears it.

        The frontend resolves the name via `ICON_CATALOG` (see
        `src/lib/icon-catalog.tsx`). Unknown names fall back to FileText
        on the client, so the server does no validation here.
        """
        patch = (
            {"$unset": {"icon": ""}}
            if icon is None
            else {"$set": {"icon": icon, "updated_at": datetime.now(timezone.utc)}}
        )
        result = await self.collection.update_one({"document_id": document_id}, patch)
        return result.modified_count > 0

    async def set_summary(self, document_id: str, summary: str | None) -> bool:
        """Persist the one-line catalog summary. Empty string clears it."""
        patch = (
            {"$unset": {"summary": ""}}
            if summary in (None, "")
            else {"$set": {"summary": summary, "updated_at": datetime.now(timezone.utc)}}
        )
        result = await self.collection.update_one({"document_id": document_id}, patch)
        return result.modified_count > 0

    async def ensure_indexes(self) -> None:
        """Indexes for slug lookup + domain filtering.

        Safe to run on every startup; Mongo no-ops duplicates.
        Non-unique on slug because the same slug can coexist across private
        user buckets; uniqueness is enforced at allocation time.
        """
        try:
            await self.collection.create_index("slug")
            await self.collection.create_index("domain_slug")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not create documents indexes: {e}")

    async def backfill_slugs(self) -> int:
        """Assign a slug to every document that doesn't have one.

        Runs once at startup; no-op on a tree that's already slug-complete.
        Returns the number of docs updated.
        """
        cursor = self.collection.find(
            {"$or": [{"slug": {"$exists": False}}, {"slug": None}, {"slug": ""}]},
            {"_id": 0, "document_id": 1, "title": 1, "filename": 1, "scope": 1, "user_id": 1},
        )
        updated = 0
        async for doc in cursor:
            title = (doc.get("title") or doc.get("filename") or doc["document_id"])
            scope = doc.get("scope") or "public"
            user_id = doc.get("user_id")
            slug = await self.generate_unique_slug(title, scope=scope, user_id=user_id)
            await self.collection.update_one(
                {"document_id": doc["document_id"]},
                {"$set": {"slug": slug}},
            )
            updated += 1
        if updated:
            logger.info(f"[backfill] documents.slug assigned to {updated} legacy records")
        return updated

    async def list_all(
        self,
        skip: int = 0,
        limit: int = 50,
        category: str | None = None,
        user_id: str | None = None,
        scope: str | None = None,
        domain_slug: str | None = None,
    ) -> list[dict]:
        """List documents with visibility filtering (Phase 14).

        - Public catalog docs have ``scope='public'`` (or no scope field on
          legacy records) and are visible to everyone.
        - Private docs have ``scope='private'`` and ``user_id=<owner>`` — only
          the owner sees them.

        ``category='workspace'`` is a UX shorthand that the API layer now
        translates to ``scope='private', user_id=<me>`` — callers that pass
        both ``category`` and ``user_id``/``scope`` still get AND semantics.
        """
        query: dict[str, Any] = {}
        if category:
            query["category"] = category
        if domain_slug:
            query["domain_slug"] = domain_slug
        if scope == "private":
            query["scope"] = "private"
            if user_id is not None:
                query["user_id"] = user_id
        elif scope == "public":
            # Public + legacy (pre-Phase-14) docs with no scope field are both visible
            query["$or"] = [{"scope": "public"}, {"scope": {"$exists": False}}]
        elif user_id is not None:
            # Caller wants to see everything visible to them: public OR owned private
            query["$or"] = [
                {"scope": "public"},
                {"scope": {"$exists": False}},
                {"scope": "private", "user_id": user_id},
            ]

        cursor = self.collection.find(query, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit)
        return [doc async for doc in cursor]

    async def delete(self, document_id: str) -> bool:
        result = await self.collection.delete_one({"document_id": document_id})
        return result.deleted_count > 0
