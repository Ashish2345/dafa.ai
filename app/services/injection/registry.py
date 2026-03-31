"""Load `sources.yaml` and construct document source instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from loguru import logger

from app.services.injection.sources.base import BaseDocumentSource
from app.services.injection.sources.finance_acts import FinanceActsSource
from app.services.injection.sources.gazette import GazetteSource
from app.services.injection.sources.ird import IrdSource
from app.services.injection.sources.lawcommission import LawCommissionSource
from app.services.injection.sources.local import LocalDirectorySource
from app.services.injection.sources.nrb import NrbSource
from app.settings import Settings

PROVIDER_MAP: Dict[str, type[BaseDocumentSource]] = {
    "lawcommission": LawCommissionSource,
    "finance_acts": FinanceActsSource,
    "nrb": NrbSource,
    "ird": IrdSource,
    "gazette": GazetteSource,
    "local": LocalDirectorySource,
}


def resolve_sources_config_path(settings: Settings) -> Path:
    p = Path(settings.injection_sources_config)
    if p.is_absolute():
        return p
    backend_root = Path(__file__).resolve().parents[3]
    return (backend_root / p).resolve()


def load_sources_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Injection sources config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


class SourceRegistry:
    """Registry of enabled document sources from YAML."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._path = resolve_sources_config_path(settings)
        self._raw = load_sources_yaml(self._path)
        self._sources_block: Dict[str, Any] = self._raw.get("sources") or {}

    def list_category_keys(self) -> List[str]:
        return list(self._sources_block.keys())

    def build_sources(
        self,
        category_filter: List[str] | None = None,
    ) -> List[Tuple[str, BaseDocumentSource]]:
        delay = float(self._settings.injection_scraper_delay)
        out: List[Tuple[str, BaseDocumentSource]] = []
        for key, block in self._sources_block.items():
            if category_filter is not None and key not in category_filter:
                continue
            if not isinstance(block, dict):
                continue
            if not block.get("enabled", True):
                logger.info("Skipping disabled injection source: {}", key)
                continue
            provider = (block.get("provider") or "").strip().lower()
            cls = PROVIDER_MAP.get(provider)
            if not cls:
                logger.warning("Unknown injection provider '{}' for category {}", provider, key)
                continue
            instance = cls(
                category_key=key,
                config=block,
                scraper_delay_s=delay,
            )
            out.append((key, instance))
        return out
