"""
Ingestion pipeline orchestrator.

WHY: Document ingestion is a multi-step process with failure modes at every
stage. A pipeline orchestrator manages the flow, handles errors gracefully,
tracks progress in the database, and ensures idempotency via content hashing.

ARCHITECTURE DECISION: Synchronous pipeline (load → process → chunk → embed
→ store) rather than async task queue because:
1. Simplicity — no Celery/Redis dependency
2. Immediate feedback — API returns chunk count on completion
3. Acceptable latency — most documents process in <30s
4. Error visibility — failures are returned directly to the caller

For large-scale batch ingestion (1000+ documents), consider adding a
background task queue (Cloud Tasks or Celery) in front of this pipeline.

TRADE-OFF: Blocking the API request during processing means large files
(>50MB PDFs) may approach timeout limits. The Makefile and Eventarc
integration handle large-scale batch ingestion separately.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.documents import Document

from src.config.logging_config import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Pipeline stages:
    1. Load: Parse raw bytes into Document objects
    2. Process: Clean text, enrich metadata, deduplicate
    3. Chunk: Split into overlapping chunks for embedding
    4. Store: Embed and upsert into Qdrant vector store
    5. Archive: Save raw document to GCS for retention

    Tracks all ingestion jobs in PostgreSQL via DocumentRecord model.
    """

    def __init__(
        self,
        settings: Settings,
        vectorstore: Any,
        embedding_manager: Any,
    ) -> None:
        self.settings = settings
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager

        # Lazy-loaded components
        self._loader: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._chunker: Optional[Any] = None
        self._storage: Optional[Any] = None

    @property
    def loader(self) -> Any:
        if self._loader is None:
            from src.ingestion.loaders.document_loader import DocumentLoader
            self._loader = DocumentLoader()
        return self._loader

    @property
    def processor(self) -> Any:
        if self._processor is None:
            from src.ingestion.processors.document_processor import (
                DocumentProcessor,
            )
            self._processor = DocumentProcessor()
        return self._processor

    @property
    def chunker(self) -> Any:
        if self._chunker is None:
            from src.retrieval.chunking import DocumentChunker
            self._chunker = DocumentChunker(self.settings)
        return self._chunker

    @property
    def storage(self) -> Any:
        if self._storage is None:
            from src.ingestion.storage.gcs_storage import GCSStorage
            self._storage = GCSStorage(self.settings)
        return self._storage

    async def process_document(
        self,
        content: bytes,
        filename: str,
        file_type: str,
        content_hash: str | None = None,
        job_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Process a single document through the full ingestion pipeline.

        Args:
            content: Raw file bytes
            filename: Original filename
            file_type: File extension (e.g., ".pdf")
            content_hash: Pre-computed SHA-256 hash (optional)
            job_id: Unique job identifier for tracking
            metadata: Additional metadata to attach

        Returns:
            Number of chunks created and stored

        Raises:
            RuntimeError: If any pipeline stage fails critically
        """
        start_time = time.perf_counter()
        content_hash = content_hash or hashlib.sha256(content).hexdigest()

        logger.info(
            "Ingestion pipeline started",
            job_id=job_id,
            filename=filename,
            file_type=file_type,
            size_bytes=len(content),
            content_hash=content_hash[:16],
        )

        try:
            # === Stage 1: Load ===
            stage_start = time.perf_counter()
            documents = self.loader.load_from_bytes(
                content=content,
                filename=filename,
                file_type=file_type,
                metadata={
                    **(metadata or {}),
                    "content_hash": content_hash,
                    "job_id": job_id,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            if not documents:
                logger.warning(
                    "No content extracted from document",
                    filename=filename,
                )
                return 0

            load_ms = (time.perf_counter() - stage_start) * 1000
            logger.info(
                "Stage 1 (Load) complete",
                pages=len(documents),
                latency_ms=round(load_ms, 2),
            )

            # === Stage 2: Process ===
            stage_start = time.perf_counter()
            processed = self.processor.process_documents(documents)
            processed = self.processor.deduplicate(processed)

            if not processed:
                logger.warning(
                    "All content filtered during processing",
                    filename=filename,
                )
                return 0

            process_ms = (time.perf_counter() - stage_start) * 1000
            logger.info(
                "Stage 2 (Process) complete",
                input_pages=len(documents),
                output_pages=len(processed),
                latency_ms=round(process_ms, 2),
            )

            # === Stage 3: Chunk ===
            stage_start = time.perf_counter()
            chunks = self.chunker.chunk_documents(processed)

            if not chunks:
                logger.warning(
                    "No chunks produced", filename=filename
                )
                return 0

            chunk_ms = (time.perf_counter() - stage_start) * 1000
            logger.info(
                "Stage 3 (Chunk) complete",
                num_chunks=len(chunks),
                avg_chunk_size=sum(len(c.page_content) for c in chunks)
                // max(len(chunks), 1),
                latency_ms=round(chunk_ms, 2),
            )

            # === Stage 4: Embed & Store ===
            stage_start = time.perf_counter()
            num_stored = await self.vectorstore.upsert_documents(
                documents=chunks,
                batch_size=self.settings.embedding.batch_size,
            )

            store_ms = (time.perf_counter() - stage_start) * 1000
            logger.info(
                "Stage 4 (Store) complete",
                num_stored=num_stored,
                latency_ms=round(store_ms, 2),
            )

            # === Stage 5: Archive to GCS (non-blocking) ===
            try:
                await self.storage.initialize()
                await self.storage.upload_raw(
                    content=content,
                    filename=filename,
                    metadata={
                        "content_hash": content_hash,
                        "job_id": job_id or "",
                        "num_chunks": str(num_stored),
                    },
                )
            except Exception as e:
                # Archive failure should not fail the pipeline
                logger.warning(
                    "Document archival failed (non-critical)",
                    error=str(e),
                )

            # === Pipeline Complete ===
            total_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Ingestion pipeline complete",
                job_id=job_id,
                filename=filename,
                total_chunks=num_stored,
                total_latency_ms=round(total_ms, 2),
                stage_latencies={
                    "load_ms": round(load_ms, 2),
                    "process_ms": round(process_ms, 2),
                    "chunk_ms": round(chunk_ms, 2),
                    "store_ms": round(store_ms, 2),
                },
            )

            return num_stored

        except Exception as e:
            total_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "Ingestion pipeline failed",
                job_id=job_id,
                filename=filename,
                error=str(e),
                latency_ms=round(total_ms, 2),
            )
            raise RuntimeError(
                f"Ingestion failed for {filename}: {e}"
            ) from e

    async def process_batch(
        self,
        files: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Process multiple documents in sequence.

        Args:
            files: List of dicts with keys: content, filename, file_type

        Returns:
            Summary dict with total_chunks, successes, failures
        """
        total_chunks = 0
        successes = 0
        failures = []

        for file_info in files:
            try:
                chunks = await self.process_document(
                    content=file_info["content"],
                    filename=file_info["filename"],
                    file_type=file_info["file_type"],
                    metadata=file_info.get("metadata"),
                )
                total_chunks += chunks
                successes += 1
            except Exception as e:
                failures.append(
                    {
                        "filename": file_info["filename"],
                        "error": str(e),
                    }
                )

        return {
            "total_files": len(files),
            "successes": successes,
            "failures": len(failures),
            "total_chunks": total_chunks,
            "failed_files": failures,
        }
