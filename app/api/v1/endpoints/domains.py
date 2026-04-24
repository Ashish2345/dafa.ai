"""
Legal-domain catalog endpoints — `GET /domains`, `GET /domains/{slug}`.

Public reads (no auth): these power Home's Browse-by-area chips and the
DomainPage, both of which render before a user authenticates in some flows.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.db.mongodb import get_database
from app.db.repositories.legal_domain_repository import LegalDomainRepository
from app.models.legal_domain import LegalDomain


router = APIRouter(prefix="/domains", tags=["domains"])


@router.get("", response_model=list[LegalDomain])
async def list_domains(db=Depends(get_database)) -> list[LegalDomain]:
    """Return all legal domains, ordered by display_order."""
    rows = await LegalDomainRepository(db).list_all()
    return [LegalDomain(**row) for row in rows]


@router.get("/{slug}", response_model=LegalDomain)
async def get_domain(slug: str, db=Depends(get_database)) -> LegalDomain:
    row = await LegalDomainRepository(db).get(slug)
    if not row:
        raise HTTPException(status_code=404, detail=f"Domain '{slug}' not found")
    return LegalDomain(**row)
