"""
Structured logging configuration with Pydantic Logfire integration.

WHY: Production systems need structured, queryable logs — not unstructured text.
Logfire provides OpenTelemetry-native structured logging with automatic span
correlation, FastAPI/SQLAlchemy instrumentation, and a hosted dashboard.

ARCHITECTURE DECISION: We layer three logging concerns:
1. structlog — structured log formatting for application logs
2. Logfire — OpenTelemetry spans for distributed tracing
3. Standard logging — compatibility bridge for third-party libraries

All three emit structured JSON in production and human-readable output in dev.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.config.settings import Settings


def setup_logging(settings: "Settings") -> None:
    """
    Configure structured logging for the application.

    Args:
        settings: Application settings instance.
    """
    log_level = getattr(logging, settings.log_level, logging.INFO)
    is_dev = settings.is_development

    # === Configure structlog ===
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if is_dev:
        # Human-readable output for development
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event=50,
        )
    else:
        # JSON output for production (machine-parseable)
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # === Configure standard library logging ===
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add structured handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    if is_dev:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy_logger in [
        "httpx",
        "httpcore",
        "uvicorn.access",
        "sqlalchemy.engine",
        "qdrant_client",
        "sentence_transformers",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # === Initialize Logfire (if token configured) ===
    if settings.logfire.token.get_secret_value():
        _setup_logfire(settings)


def _setup_logfire(settings: "Settings") -> None:
    """
    Initialize Pydantic Logfire for distributed tracing.

    Logfire provides:
    - Automatic FastAPI request spans
    - SQLAlchemy query spans
    - httpx outbound request spans
    - Custom business logic spans
    """
    try:
        import logfire

        logfire.configure(
            token=settings.logfire.token.get_secret_value(),
            project_name=settings.logfire.project_name,
            environment=settings.logfire.environment,
            send_to_logfire=True,
        )

        # Instrument libraries
        logfire.instrument_fastapi()
        logfire.instrument_sqlalchemy()
        logfire.instrument_httpx()

        logger = structlog.get_logger("logfire")
        logger.info("Logfire initialized", project=settings.logfire.project_name)

    except ImportError:
        logger = structlog.get_logger("logfire")
        logger.warning("Logfire package not installed, skipping initialization")
    except Exception as e:
        logger = structlog.get_logger("logfire")
        logger.error("Failed to initialize Logfire", error=str(e))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing query", query_id="abc123", user_id="user1")
    """
    return structlog.get_logger(name)
