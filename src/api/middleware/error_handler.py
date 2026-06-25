"""
Global exception handlers for the FastAPI application.

WHY: Unhandled exceptions should never leak stack traces or internal details
to clients. This middleware catches all exceptions, logs them with incident
IDs, and returns sanitized error responses.
"""

from __future__ import annotations

import traceback
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class AgentError(Exception):
    """Custom exception for agent pipeline errors."""

    def __init__(self, message: str, details: str | None = None) -> None:
        self.message = message
        self.details = details
        super().__init__(self.message)


class IngestionError(Exception):
    """Custom exception for document ingestion errors."""

    def __init__(self, message: str, filename: str | None = None) -> None:
        self.message = message
        self.filename = filename
        super().__init__(self.message)


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(ValidationError)
    async def validation_error_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors — 422."""
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(AgentError)
    async def agent_error_handler(
        request: Request, exc: AgentError
    ) -> JSONResponse:
        """Handle agent pipeline errors — 500."""
        incident_id = str(uuid.uuid4())[:8]
        logger.error(
            "Agent error",
            incident_id=incident_id,
            message=exc.message,
            details=exc.details,
            path=str(request.url),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Agent Processing Error",
                "detail": exc.message,
                "incident_id": incident_id,
            },
        )

    @app.exception_handler(IngestionError)
    async def ingestion_error_handler(
        request: Request, exc: IngestionError
    ) -> JSONResponse:
        """Handle ingestion errors — 400."""
        return JSONResponse(
            status_code=400,
            content={
                "error": "Ingestion Error",
                "detail": exc.message,
                "filename": exc.filename,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all for unhandled exceptions — 500.

        Logs the full traceback but returns a sanitized message to the client.
        """
        incident_id = str(uuid.uuid4())[:8]
        logger.error(
            "Unhandled exception",
            incident_id=incident_id,
            error=str(exc),
            traceback=traceback.format_exc(),
            path=str(request.url),
            method=request.method,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred. Please try again.",
                "incident_id": incident_id,
            },
        )
