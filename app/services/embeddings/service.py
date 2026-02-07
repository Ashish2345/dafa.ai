"""
Embedding Service

Generates embeddings for text chunks using OpenAI.
"""

from typing import List, Optional

from loguru import logger
from openai import OpenAI

from app.settings import settings


class EmbeddingService:
    """
    Service for generating embeddings from text chunks.

    Uses OpenAI's text-embedding-3-small model for generating embeddings.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
        """
        Initialize the embedding service.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY from settings)
            model: Embedding model to use (default: text-embedding-3-small)
        """
        self.api_key = api_key or getattr(settings, "openai_api_key", None)
        if not self.api_key:
            logger.warning("OpenAI API key not found. Embedding generation will fail.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors (each is a list of floats)

        Raises:
            ValueError: If OpenAI API key is not configured
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env file.")

        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts using {self.model}")

        all_embeddings = []

        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Processing embedding batch {i // batch_size + 1} ({len(batch)} texts)")

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Return empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batch])

        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector (list of floats)
        """
        embeddings = self.generate_embeddings([text], batch_size=1)
        return embeddings[0] if embeddings else []
