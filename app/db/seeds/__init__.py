"""Startup-time seeds for catalog / reference data."""

from app.db.seeds.legal_domains_seed import seed_legal_domains

__all__ = ["seed_legal_domains"]
