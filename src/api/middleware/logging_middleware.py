"""
Request/response logging middleware with correlation IDs.

WHY: Every request needs a unique correlation ID for tracing across
distributed services. This middleware generates the ID, attaches it to
logs and response headers, and records request/response metadata.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware for request/response logging with correlation IDs.

    Attaches a unique X-Request-ID to every request, logs timing,
    and propagates the ID in response headers for client-side tracing.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate or reuse correlation ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

        # Bind correlation ID to structlog context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
        )

        start_time = time.perf_counter()

        # Log incoming request
        logger.info(
            "Request started",
            client=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("User-Agent", "")[:100],
        )

        try:
            response = await call_next(request)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Add correlation headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{latency_ms:.2f}ms"

            # Log response
            logger.info(
                "Request completed",
                status_code=response.status_code,
                latency_ms=round(latency_ms, 2),
            )

            return response

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Request failed",
                error=str(e),
                latency_ms=round(latency_ms, 2),
            )
            raise
