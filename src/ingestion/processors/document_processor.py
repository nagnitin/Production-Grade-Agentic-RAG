"""
Document text processing and cleaning pipeline.

WHY: Raw extracted text is noisy — it contains control characters, excessive
whitespace, headers/footers, page numbers, and encoding artifacts. Clean text
produces better embeddings and more accurate retrieval.

ARCHITECTURE DECISION: Processing as a pipeline of composable transformations.
Each step is a pure function that takes text and returns cleaned text.
This makes it easy to add, remove, or reorder cleaning steps.

TRADE-OFF: Aggressive cleaning may remove meaningful formatting (e.g., code
indentation, tables). We preserve paragraph structure and only strip
obvious noise. For code-heavy documents, consider disabling whitespace
normalization.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

from langchain_core.documents import Document

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Text cleaning and metadata enrichment pipeline.

    Pipeline steps (applied in order):
    1. Unicode normalization (NFKC)
    2. Control character removal
    3. Whitespace normalization
    4. Header/footer stripping
    5. Minimum content validation
    6. Metadata enrichment
    """

    # Minimum content length (characters) to keep a document page
    MIN_CONTENT_LENGTH = 50

    # Common header/footer patterns to strip
    NOISE_PATTERNS = [
        # Page numbers
        r"^\s*-?\s*\d+\s*-?\s*$",
        # "Page X of Y"
        r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",
        # Repeated characters (separators)
        r"^[=\-_~*]{10,}$",
        # Confidential notices
        r"^\s*confidential\s*$",
        # "All rights reserved"
        r"^\s*all\s+rights?\s+reserved\.?\s*$",
    ]

    def __init__(self) -> None:
        self._noise_regex = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.NOISE_PATTERNS
        ]

    def process_documents(
        self,
        documents: list[Document],
        min_length: int | None = None,
    ) -> list[Document]:
        """
        Clean and enrich a list of documents.

        Args:
            documents: Raw documents from loaders
            min_length: Override minimum content length

        Returns:
            Processed documents with cleaned text and enriched metadata
        """
        min_len = min_length or self.MIN_CONTENT_LENGTH
        processed = []

        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)

            # Skip documents with insufficient content
            if len(cleaned_text.strip()) < min_len:
                logger.debug(
                    "Skipping short document",
                    length=len(cleaned_text),
                    source=doc.metadata.get("source", "unknown"),
                )
                continue

            # Enrich metadata
            enriched_metadata = self._enrich_metadata(
                cleaned_text, doc.metadata
            )

            processed.append(
                Document(
                    page_content=cleaned_text,
                    metadata=enriched_metadata,
                )
            )

        logger.info(
            "Document processing complete",
            input_docs=len(documents),
            output_docs=len(processed),
            filtered_out=len(documents) - len(processed),
        )

        return processed

    def clean_text(self, text: str) -> str:
        """
        Apply the full cleaning pipeline to a text string.

        Steps applied in order for maximum effectiveness.
        """
        if not text:
            return ""

        # 1. Unicode normalization (NFKC combines compatibility chars)
        text = unicodedata.normalize("NFKC", text)

        # 2. Remove control characters (keep newlines, tabs, spaces)
        text = self._remove_control_chars(text)

        # 3. Remove noise patterns (headers, footers, page numbers)
        text = self._remove_noise_patterns(text)

        # 4. Normalize whitespace
        text = self._normalize_whitespace(text)

        # 5. Strip leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """Remove non-printable control characters, preserving newlines and tabs."""
        cleaned = []
        for char in text:
            if char in ("\n", "\r", "\t"):
                cleaned.append(char)
            elif unicodedata.category(char).startswith("C"):
                # Control character — skip
                continue
            else:
                cleaned.append(char)
        return "".join(cleaned)

    def _remove_noise_patterns(self, text: str) -> str:
        """Remove common header/footer noise patterns line-by-line."""
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            is_noise = False
            for pattern in self._noise_regex:
                if pattern.match(line.strip()):
                    is_noise = True
                    break

            if not is_noise:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace while preserving paragraph structure.

        - Collapse multiple spaces/tabs into single space
        - Collapse 3+ newlines into double newline (paragraph break)
        - Normalize line endings to \n
        """
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse multiple spaces/tabs on the same line
        text = re.sub(r"[^\S\n]+", " ", text)

        # Collapse 3+ consecutive newlines into 2 (preserve paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    @staticmethod
    def _enrich_metadata(
        text: str, existing_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Add computed metadata fields to a document.

        Enrichments:
        - word_count: Approximate word count
        - char_count: Character count
        - content_hash: SHA-256 hash for deduplication
        - processed_at: ISO timestamp
        - doc_type: Inferred from file extension
        """
        word_count = len(text.split())
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        enriched = {
            **existing_metadata,
            "word_count": word_count,
            "char_count": len(text),
            "content_hash": content_hash,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

        # Infer doc_type from file_type if not already set
        if "doc_type" not in enriched:
            file_type = enriched.get("file_type", "")
            type_map = {
                ".pdf": "pdf",
                ".docx": "word",
                ".pptx": "presentation",
                ".html": "html",
                ".htm": "html",
                ".txt": "text",
            }
            enriched["doc_type"] = type_map.get(file_type, "unknown")

        return enriched

    @staticmethod
    def deduplicate(
        documents: list[Document],
    ) -> list[Document]:
        """
        Remove duplicate documents based on content hash.

        Uses the content_hash metadata field for O(n) deduplication.
        """
        seen_hashes: set[str] = set()
        unique = []

        for doc in documents:
            content_hash = doc.metadata.get(
                "content_hash",
                hashlib.sha256(doc.page_content.encode()).hexdigest(),
            )

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique.append(doc)

        if len(documents) != len(unique):
            logger.info(
                "Deduplicated documents",
                original=len(documents),
                unique=len(unique),
                removed=len(documents) - len(unique),
            )

        return unique
