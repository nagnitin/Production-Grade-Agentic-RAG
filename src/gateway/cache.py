"""
Semantic cache using Qdrant for repeated query detection.

WHY: Many enterprise users ask similar questions. Caching reduces latency
from seconds to milliseconds and eliminates LLM costs for cached responses.

ARCHITECTURE DECISION: Semantic caching (embedding similarity) over exact
string matching because users phrase the same question differently.
"What is our revenue?" and "How much revenue did we make?" should cache-hit.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.config.logging_config import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


class SemanticCache:
    """
    Embedding-based semantic cache for query responses.

    Stores (query_embedding, response) pairs in a dedicated Qdrant collection.
    On cache lookup, finds the most similar cached query and returns its
    response if similarity exceeds the threshold.
    """

    def __init__(
        self,
        settings: Settings,
        embedding_manager: Any,
    ) -> None:
        self.settings = settings
        self.embedding_manager = embedding_manager
        self.collection_name = settings.qdrant.collection_cache
        self.threshold = settings.cache.threshold
        self.ttl = settings.cache.ttl
        self.enabled = settings.cache.enabled
        self._client: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize the cache collection in Qdrant."""
        if not self.enabled:
            logger.info("Semantic cache disabled")
            return

        from qdrant_client import AsyncQdrantClient, models

        connection_params = self.settings.qdrant.connection_params
        self._client = AsyncQdrantClient(**connection_params)

        # Ensure cache collection exists
        try:
            collections = await self._client.get_collections()
            existing = {c.name for c in collections.collections}

            if self.collection_name not in existing:
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.settings.embedding.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info("Created cache collection", name=self.collection_name)
        except Exception as e:
            logger.error("Failed to initialize semantic cache", error=str(e))
            self.enabled = False

    async def get(self, query: str) -> Optional[dict[str, Any]]:
        """
        Look up a cached response for a semantically similar query.

        Returns None on cache miss.
        """
        if not self.enabled or not self._client:
            return None

        try:
            query_embedding = await self.embedding_manager.embed_query(query)

            results = await self._client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=1,
                score_threshold=self.threshold,
            )

            if results.points:
                cached = results.points[0]
                payload = cached.payload or {}

                # Check TTL
                cached_at = payload.get("cached_at", "")
                if cached_at:
                    cached_time = datetime.fromisoformat(cached_at)
                    age_seconds = (
                        datetime.now(timezone.utc) - cached_time
                    ).total_seconds()
                    if age_seconds > self.ttl:
                        logger.info("Cache entry expired", age_seconds=age_seconds)
                        return None

                logger.info(
                    "Cache hit",
                    similarity=round(cached.score, 4),
                    cached_query=payload.get("query", "")[:50],
                )

                return {
                    "response": payload.get("response", ""),
                    "citations": json.loads(payload.get("citations", "[]")),
                    "confidence": payload.get("confidence", 0.0),
                    "cached": True,
                    "cache_similarity": round(cached.score, 4),
                }

            return None

        except Exception as e:
            logger.warning("Cache lookup failed", error=str(e))
            return None

    async def set(
        self,
        query: str,
        response: str,
        citations: list[dict] | None = None,
        confidence: float = 0.0,
    ) -> None:
        """Store a query-response pair in the cache."""
        if not self.enabled or not self._client:
            return

        try:
            import uuid
            from qdrant_client import models

            query_embedding = await self.embedding_manager.embed_query(query)

            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=query_embedding,
                payload={
                    "query": query,
                    "response": response,
                    "citations": json.dumps(citations or []),
                    "confidence": confidence,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            await self._client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            logger.info("Cached response", query=query[:50])

        except Exception as e:
            logger.warning("Failed to cache response", error=str(e))

    async def clear(self) -> None:
        """Clear the entire cache."""
        if not self._client:
            return

        try:
            from qdrant_client import models

            await self._client.delete_collection(self.collection_name)
            logger.info("Cache cleared")
            await self.initialize()  # Recreate collection
        except Exception as e:
            logger.error("Failed to clear cache", error=str(e))

    async def close(self) -> None:
        """Close the cache client."""
        if self._client:
            await self._client.close()
