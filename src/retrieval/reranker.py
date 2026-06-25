"""
FlashRank reranker wrapper.

WHY: Cross-encoder reranking dramatically improves retrieval precision.
FlashRank runs locally (~50ms for 20 docs), eliminating API costs and latency.

ARCHITECTURE DECISION: ms-marco-MiniLM-L-12-v2 model provides strong
reranking quality at minimal compute cost. The model is ~30MB and runs
efficiently on CPU.
"""

from __future__ import annotations

from typing import Any, Optional

from src.config.logging_config import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


class FlashRankReranker:
    """
    Cross-encoder reranker using FlashRank.

    Scores query-document pairs using a cross-encoder model,
    providing fine-grained relevance ranking.
    """

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.reranker.model
        self.default_top_n = settings.reranker.top_n
        self.score_threshold = settings.reranker.score_threshold
        self._ranker: Optional[object] = None

    def _load_ranker(self) -> object:
        """Lazy-load the FlashRank ranker."""
        if self._ranker is None:
            from flashrank import Ranker

            logger.info("Loading FlashRank model", model=self.model_name)
            self._ranker = Ranker(model_name=self.model_name, cache_dir="/tmp/flashrank")
            logger.info("FlashRank model loaded")

        return self._ranker

    @property
    def ranker(self) -> object:
        return self._load_ranker()

    def rerank(
        self,
        query: str,
        passages: list[dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank passages against a query.

        Args:
            query: The search query
            passages: List of dicts with "id", "text", and optional "meta" keys
            top_n: Number of top results to return

        Returns:
            Sorted list of passages with added "score" field
        """
        from flashrank import RerankRequest

        if not passages:
            return []

        top_n = top_n or self.default_top_n

        rerank_request = RerankRequest(
            query=query,
            passages=passages,
        )

        results = self.ranker.rerank(rerank_request)

        # Convert to list of dicts with scores
        scored = []
        for result in results:
            scored.append({
                "id": result.get("id", 0),
                "text": result.get("text", ""),
                "score": result.get("score", 0.0),
                "meta": result.get("meta", {}),
            })

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        return scored[:top_n]

    def health_check(self) -> bool:
        """Verify the reranker model is loaded."""
        try:
            _ = self.ranker
            return True
        except Exception:
            return False
