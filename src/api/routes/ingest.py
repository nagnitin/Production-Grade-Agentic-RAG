"""
Ingest route — POST /ingest endpoint.

Handles document upload and triggers the ingestion pipeline.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Request, UploadFile

from src.api.middleware.auth import verify_api_key
from src.api.middleware.error_handler import IngestionError
from src.api.schemas.common import IngestResponse
from src.config.constants import MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS
from src.config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload and ingest a document",
    description="Upload a document (PDF, DOCX, PPTX, HTML, TXT) for processing and vector storage.",
)
async def ingest_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Document file to ingest"),
    _api_key: str = Depends(verify_api_key),
) -> IngestResponse:
    """Upload and process a document into the vector store."""
    job_id = str(uuid.uuid4())

    # Validate file
    if not file.filename:
        raise IngestionError("No filename provided")

    extension = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if extension not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type: {extension}. Supported: {SUPPORTED_EXTENSIONS}",
            filename=file.filename,
        )

    # Read file content
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise IngestionError(
            f"File too large: {file_size_mb:.1f}MB. Maximum: {MAX_FILE_SIZE_MB}MB",
            filename=file.filename,
        )

    content_hash = hashlib.sha256(content).hexdigest()

    logger.info(
        "Document received for ingestion",
        job_id=job_id,
        filename=file.filename,
        size_mb=round(file_size_mb, 2),
        file_type=extension,
        content_hash=content_hash[:16],
    )

    # Get ingestion pipeline from app state
    pipeline = request.app.state.ingestion_pipeline

    # Run ingestion asynchronously
    try:
        # Clear existing collection first to ensure only the active/uploaded document is queried
        vectorstore = request.app.state.vectorstore
        try:
            logger.info("Clearing vector store collection before ingesting new document")
            await vectorstore.client.delete_collection(vectorstore.collection_name)
            await vectorstore.initialize()
        except Exception as e:
            logger.warning("Could not clear collection before ingestion", error=str(e))

        num_chunks = await pipeline.process_document(
            content=content,
            filename=file.filename,
            file_type=extension,
            content_hash=content_hash,
            job_id=job_id,
        )

        # Clear semantic cache since the knowledge base has changed
        semantic_cache = getattr(request.app.state, "semantic_cache", None)
        if semantic_cache:
            logger.info("Clearing semantic cache after document ingestion")
            await semantic_cache.clear()

        metrics_collector = getattr(request.app.state, "metrics_collector", None)
        if metrics_collector:
            metrics_collector.record_ingestion(success=True)

        return IngestResponse(
            job_id=job_id,
            status="completed",
            filename=file.filename,
            file_type=extension,
            message=f"Document processed successfully. {num_chunks} chunks created.",
            num_chunks=num_chunks,
        )

    except Exception as e:
        logger.error(
            "Ingestion failed",
            job_id=job_id,
            filename=file.filename,
            error=str(e),
        )
        metrics_collector = getattr(request.app.state, "metrics_collector", None)
        if metrics_collector:
            metrics_collector.record_ingestion(success=False, error_type="processing")
        raise IngestionError(
            f"Failed to process document: {str(e)}",
            filename=file.filename,
        )


@router.post(
    "/ingest/clear",
    summary="Clear all ingested documents",
    description="Deletes all vectors and documents from the knowledge base.",
)
async def clear_ingestion_endpoint(
    request: Request,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, str]:
    """Delete all documents and vectors from Qdrant."""
    vectorstore = request.app.state.vectorstore
    try:
        await vectorstore.client.delete_collection(vectorstore.collection_name)
        await vectorstore.initialize()  # Recreate
        
        # Clear semantic cache too
        semantic_cache = getattr(request.app.state, "semantic_cache", None)
        if semantic_cache:
            await semantic_cache.clear()
            
        logger.info("Successfully cleared all ingested documents and cache")
        return {"message": "All ingested documents and cache have been successfully cleared."}
    except Exception as e:
        logger.error("Failed to clear ingested documents", error=str(e))
        raise IngestionError(f"Failed to clear documents: {str(e)}")

