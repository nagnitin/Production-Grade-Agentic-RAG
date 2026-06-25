"""
Observability Metrics module.

WHY: High-fidelity monitoring is required for enterprise scale workloads. This module
provides a thread-safe MetricsCollector that tracks query counts, ingestion rates,
cache hit rate, avg latency, and token usages. It registers Prometheus metrics
and compiles JSON summary reports.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Set

from prometheus_client import Counter, Gauge, Histogram

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Thread-safe collector for application metrics.
    
    Exposes raw metrics for Prometheus scraping and aggregates them
    for the local API summary endpoints.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._lock = threading.Lock()

        # In-memory accumulators for JSON summary endpoint
        self._total_queries = 0
        self._total_ingestions = 0
        self._total_latency_ms = 0.0
        self._latency_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_confidence = 0.0
        self._confidence_count = 0
        self._total_errors = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._active_sessions: Set[str] = set()

        # Prometheus metric definitions
        self.prom_queries = Counter(
            "rag_queries_total", "Total user queries submitted"
        )
        self.prom_ingestions = Counter(
            "rag_ingestions_total", 
            "Total ingestion runs", 
            ["status"]
        )
        self.prom_latency = Histogram(
            "rag_query_latency_seconds", 
            "Query request latency in seconds"
        )
        self.prom_cache_hits = Counter(
            "rag_cache_hits_total", 
            "Total query semantic cache hits"
        )
        self.prom_cache_misses = Counter(
            "rag_cache_misses_total", 
            "Total query semantic cache misses"
        )
        self.prom_confidence = Histogram(
            "rag_query_confidence", 
            "Agent response confidence score"
        )
        self.prom_errors = Counter(
            "rag_errors_total", 
            "Total exception count", 
            ["type"]
        )
        self.prom_tokens = Counter(
            "rag_tokens_total", 
            "Token consumption breakdown", 
            ["type"]
        )
        self.prom_active_sessions = Gauge(
            "rag_active_sessions", 
            "Current number of unique active session IDs"
        )

    def record_query(
        self, 
        latency_ms: float, 
        confidence: float, 
        cache_hit: bool, 
        error: bool = False
    ) -> None:
        """Record statistics for a single query request."""
        with self._lock:
            self._total_queries += 1
            self.prom_queries.inc()

            if error:
                self._total_errors += 1
                self.prom_errors.labels(type="query").inc()
            else:
                self._total_latency_ms += latency_ms
                self._latency_count += 1
                self.prom_latency.observe(latency_ms / 1000.0)

                self._total_confidence += confidence
                self._confidence_count += 1
                self.prom_confidence.observe(confidence)

                if cache_hit:
                    self._cache_hits += 1
                    self.prom_cache_hits.inc()
                else:
                    self._cache_misses += 1
                    self.prom_cache_misses.inc()

    def record_ingestion(self, success: bool = True, error_type: str = "") -> None:
        """Record statistics for a single file ingestion pipeline run."""
        with self._lock:
            self._total_ingestions += 1
            if success:
                self.prom_ingestions.labels(status="success").inc()
            else:
                self._total_errors += 1
                self.prom_ingestions.labels(status="failed").inc()
                self.prom_errors.labels(type=f"ingestion_{error_type}").inc()

    def record_tokens(self, prompt: int, completion: int) -> None:
        """Record input and output token consumption."""
        with self._lock:
            self._prompt_tokens += prompt
            self._completion_tokens += completion
            self._total_tokens += (prompt + completion)

            self.prom_tokens.labels(type="prompt").inc(prompt)
            self.prom_tokens.labels(type="completion").inc(completion)
            self.prom_tokens.labels(type="total").inc(prompt + completion)

    def record_session(self, session_id: str) -> None:
        """Record session interaction."""
        with self._lock:
            self._active_sessions.add(session_id)
            self.prom_active_sessions.set(len(self._active_sessions))

    def get_summary(self) -> dict[str, Any]:
        """Compile a summary of current metric metrics."""
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._latency_count
                if self._latency_count > 0
                else 0.0
            )
            total_cache_ops = self._cache_hits + self._cache_misses
            cache_hit_rate = (
                self._cache_hits / total_cache_ops
                if total_cache_ops > 0
                else 0.0
            )
            avg_confidence = (
                self._total_confidence / self._confidence_count
                if self._confidence_count > 0
                else 0.0
            )
            error_rate = (
                self._total_errors / self._total_queries
                if self._total_queries > 0
                else 0.0
            )

            return {
                "total_queries": self._total_queries,
                "total_ingestions": self._total_ingestions,
                "avg_latency_ms": round(avg_latency, 2),
                "cache_hit_rate": round(cache_hit_rate, 4),
                "avg_confidence": round(avg_confidence, 4),
                "error_rate": round(error_rate, 4),
                "token_usage": {
                    "prompt_tokens": self._prompt_tokens,
                    "completion_tokens": self._completion_tokens,
                    "total_tokens": self._total_tokens,
                },
                "active_sessions": len(self._active_sessions),
            }
