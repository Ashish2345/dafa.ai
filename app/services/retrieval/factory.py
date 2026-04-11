"""
Retrieval strategy factory.

Selects the right strategy based on request param or system default.
"""

from app.db.mongodb import get_database
from app.db.repositories.page_index_repository import PageIndexRepository
from app.services.retrieval.base import RetrievalStrategy
from app.settings import settings


class RetrievalFactory:
    @staticmethod
    async def get_strategy(strategy_name: str | None = None) -> RetrievalStrategy:
        """
        Return the appropriate RetrievalStrategy.

        Args:
            strategy_name: "page_index" or "vector". Defaults to settings.default_retrieval_strategy.
        """
        name = strategy_name or settings.default_retrieval_strategy

        if name == "page_index":
            from app.services.retrieval.page_index.strategy import PageIndexStrategy
            db = await get_database()
            repo = PageIndexRepository(db)
            return PageIndexStrategy(repository=repo)

        elif name == "vector":
            from app.services.retrieval.vector.strategy import VectorStrategy
            return VectorStrategy()

        raise ValueError(f"Unknown strategy: {name!r}. Use 'page_index' or 'vector'.")
