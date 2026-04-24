"""
Idempotent seed for the `legal_domains` catalog.

Run at app startup. Safe to run repeatedly — uses upsert on `slug`.
Operators can override any row afterwards via `PATCH /domains/{slug}`
(coming when admin surfaces are built); the seed never clobbers
existing values because it doesn't pass `$setOnInsert` for mutable
fields — it writes through every field. If you need to preserve
operator edits, switch this to idempotent-only (insert-if-missing).
"""

from __future__ import annotations

from loguru import logger

from app.db.repositories.legal_domain_repository import LegalDomainRepository


DEFAULT_DOMAINS: list[dict] = [
    {
        "slug": "tax",
        "name": "Tax & Revenue",
        "icon": "Banknote",
        "icon_color_bg": "#f1e7d0",
        "icon_color_fg": "#6b4f1c",
        "description": "Income Tax, VAT, Excise, Customs, and related rules.",
        "display_order": 10,
    },
    {
        "slug": "company",
        "name": "Company & Securities",
        "icon": "Building2",
        "icon_color_bg": "#dedbed",
        "icon_color_fg": "#3a346b",
        "description": "Companies Act, SEBON regulations, insolvency law.",
        "display_order": 20,
    },
    {
        "slug": "banking",
        "name": "Banking & NRB",
        "icon": "Landmark",
        "icon_color_bg": "#d8e1ea",
        "icon_color_fg": "#22384e",
        "description": "BAFIA, NRB Act, directives, foreign exchange.",
        "display_order": 30,
    },
    {
        "slug": "labor",
        "name": "Labor & HR",
        "icon": "Briefcase",
        "icon_color_bg": "#ebd8cc",
        "icon_color_fg": "#6c3620",
        "description": "Labor Act, Social Security Fund, bonus & trade union.",
        "display_order": 40,
    },
    {
        "slug": "criminal",
        "name": "Criminal & Civil",
        "icon": "Scale",
        "icon_color_bg": "#ebd3cd",
        "icon_color_fg": "#6b2418",
        "description": "Muluki Civil Code, Criminal Code, procedure codes.",
        "display_order": 50,
    },
    {
        "slug": "judgments",
        "name": "Judgments & Najirs",
        "icon": "Gavel",
        "icon_color_bg": "#dedbc6",
        "icon_color_fg": "#4a451c",
        "description": "Supreme Court precedents, landmark decisions.",
        "display_order": 60,
    },
    {
        "slug": "gazettes",
        "name": "Gazettes",
        "icon": "Newspaper",
        "icon_color_bg": "#d9e1d3",
        "icon_color_fg": "#2e4a27",
        "description": "Raw Nepal Gazette publication stream — notices, amendments.",
        "display_order": 70,
    },
    {
        "slug": "constitutional",
        "name": "Constitutional",
        "icon": "BookOpen",
        "icon_color_bg": "#d4d9e3",
        "icon_color_fg": "#1e2a44",
        "description": "Constitution of Nepal, directive principles, rights.",
        "display_order": 80,
    },
]


async def seed_legal_domains(db) -> None:
    """Upsert the eight default legal domains into Mongo.

    Called from the FastAPI startup hook. Blocks briefly on app boot.
    """
    repo = LegalDomainRepository(db)
    await repo.ensure_indexes()
    for row in DEFAULT_DOMAINS:
        await repo.upsert(row)
    logger.info(f"[seed] legal_domains: ensured {len(DEFAULT_DOMAINS)} rows")
