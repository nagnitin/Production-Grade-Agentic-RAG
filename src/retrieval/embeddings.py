"""
Embedding manager using HuggingFace sentence-transformers.

WHY: Embeddings are the foundation of semantic search. Using a local model
(BAAI/bge-base-en-v1.5) eliminates API costs, reduces latency, and removes
external dependencies. The model runs on CPU within Cloud Run containers.

ARCHITECTURE DECISION: bge-base-en-v1.5 (768 dims) over alternatives:
- bge-large: Better quality but 2x slower, 2x memory. Marginal improvement
  with FlashRank reranking in the pipeline.
- OpenAI ada-002: Requires API calls, adds latency and cost.
- bge-small: Faster but measurably worse retrieval quality.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Optional

import numpy as np

from src.config.logging_config import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


class EmbeddingManager:
    """
    Manages embedding model lifecycle and inference.

    Features:
    - Lazy model loading (loaded on first use)
    - Thread-safe async wrapping of sync model
    - Batch embedding with configurable batch size
    - Model caching via singleton pattern
    """

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.embedding.model
        self.dimension = settings.embedding.dimension
        self.batch_size = settings.embedding.batch_size
        self.device = settings.embedding.device
        self.normalize = settings.embedding.normalize
        self._model: Optional[object] = None

    def _load_model(self) -> object:
        """Load the sentence-transformers model (lazy)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model",
                model=self.model_name,
                device=self.device,
            )

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

            logger.info("Embedding model loaded", model=self.model_name)

        return self._model

    @property
    def model(self) -> object:
        return self._load_model()

    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        For query embedding, bge models expect a specific prefix.
        """
        loop = asyncio.get_event_loop()

        def _embed():
            # BGE models use "Represent this sentence: " prefix for queries
            prefixed = f"Represent this sentence: {text}" if "bge" in self.model_name.lower() else text
            embedding = self.model.encode(
                prefixed,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return embedding.tolist()

        return await loop.run_in_executor(None, _embed)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document texts in batches.

        Document embeddings don't use the query prefix for bge models.
        """
        if not texts:
            return []

        loop = asyncio.get_event_loop()

        def _embed_batch():
            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                )
                all_embeddings.extend(embeddings.tolist())

            return all_embeddings

        return await loop.run_in_executor(None, _embed_batch)

    async def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embeddings = await self.embed_documents([text1, text2])
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])

        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(cosine_sim)

    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.dimension

    def get_model(self) -> object:
        """Return a LangChain-compatible HuggingFaceEmbeddings model."""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device}
        )

    async def health_check(self) -> bool:
        """Verify the model can produce embeddings."""
        try:
            result = await self.embed_query("health check")
            return len(result) == self.dimension
        except Exception:
            return False
