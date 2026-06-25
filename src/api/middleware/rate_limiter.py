"""
Rate limiter middleware using token bucket algorithm.

WHY: Rate limiting prevents abuse, ensures fair resource allocation, and
protects downstream services (LLM APIs, databases) from overload.

ARCHITECTURE DECISION: In-memory token bucket for simplicity. For multi-instance
deployments, swap to Redis-backed (e.g., redis-py with Lua scripts).
The middleware is a Starlette-compatible ASGI middleware.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from fastapi import HTTPException, Request

from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


class TokenBucket:
    """Token bucket rate limiter per key."""

    def __init__(self, rate: int, window: int) -> None:
        self.rate = rate          # Tokens per window
        self.window = window      # Window in seconds
        self.tokens: dict[str, float] = defaultdict(lambda: float(rate))
        self.last_refill: dict[str, float] = defaultdict(time.time)

    def consume(self, key: str) -> tuple[bool, float]:
        """
        Try to consume a token. Returns (allowed, retry_after).
        """
        now = time.time()
        elapsed = now - self.last_refill[key]

        # Refill tokens based on elapsed time
        self.tokens[key] = min(
            self.rate,
            self.tokens[key] + (elapsed * self.rate / self.window),
        )
        self.last_refill[key] = now

        if self.tokens[key] >= 1:
            self.tokens[key] -= 1
            return True, 0.0
        else:
            # Calculate retry-after
            retry_after = (1 - self.tokens[key]) * self.window / self.rate
            return False, retry_after


# Global rate limiter instance
_rate_limiter: TokenBucket | None = None


def get_rate_limiter() -> TokenBucket:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = TokenBucket(
            rate=settings.rate_limit.requests,
            window=settings.rate_limit.window,
        )
    return _rate_limiter


async def rate_limit_dependency(request: Request) -> None:
    """
    FastAPI dependency for rate limiting.

    Rate limits by API key (from x-api-key header) or by client IP.
    """
    # Skip rate limiting for health and metrics endpoints
    if request.url.path in ("/health", "/metrics", "/docs", "/openapi.json"):
        return

    # Use API key as rate limit key, fallback to IP
    key = request.headers.get("x-api-key", request.client.host if request.client else "unknown")

    limiter = get_rate_limiter()
    allowed, retry_after = limiter.consume(key)

    if not allowed:
        logger.warning(
            "Rate limit exceeded",
            key=key[:16] + "...",
            retry_after=round(retry_after, 1),
        )
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
