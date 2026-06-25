"""
Document chunking strategies.

WHY: LLMs have context windows. Documents must be split into chunks that:
1. Fit within the embedding model's token limit (~512 tokens for bge)
2. Preserve semantic coherence (don't split mid-paragraph)
3. Maintain enough context for accurate retrieval
4. Include overlap to prevent information loss at boundaries

ARCHITECTURE DECISION: Two strategies available:
1. RecursiveCharacterTextSplitter — Default. Reliable, fast, predictable.
   Splits on paragraph → sentence → word boundaries.
2. SemanticChunker — Experimental. Uses embeddings to find natural topic
   boundaries. Better coherence but slower and less predictable sizes.

We default to recursive splitting because it's battle-tested in production.
Semantic chunking is available for users who need it.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from src.config.logging_config import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


class DocumentChunker:
    """
    Multi-strategy document chunker.

    Splits documents into overlapping chunks suitable for embedding
    and vector storage, preserving metadata through the split.
    """

    def __init__(self, settings: Settings) -> None:
        self.chunk_size = settings.chunking.size
        self.chunk_overlap = settings.chunking.overlap
        self.strategy = settings.chunking.strategy

    def chunk_documents(
        self,
        documents: list[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> list[Document]:
        """
        Split documents into chunks using the configured strategy.

        Args:
            documents: List of LangChain Documents to chunk
            chunk_size: Override default chunk size
            chunk_overlap: Override default overlap
            strategy: Override default strategy ("recursive" or "semantic")

        Returns:
            List of chunked Documents with preserved and enriched metadata
        """
        size = chunk_size or self.chunk_size
        overlap = chunk_overlap or self.chunk_overlap
        strat = strategy or self.strategy

        logger.info(
            "Chunking documents",
            num_docs=len(documents),
            strategy=strat,
            chunk_size=size,
            overlap=overlap,
        )

        if strat == "semantic":
            chunks = self._semantic_chunk(documents, size)
        else:
            chunks = self._recursive_chunk(documents, size, overlap)

        # Enrich chunk metadata
        enriched = self._enrich_metadata(chunks, documents)

        logger.info(
            "Chunking complete",
            input_docs=len(documents),
            output_chunks=len(enriched),
            avg_chunk_size=sum(len(c.page_content) for c in enriched) // max(len(enriched), 1),
        )

        return enriched

    def _recursive_chunk(
        self,
        documents: list[Document],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]:
        """
        Split using RecursiveCharacterTextSplitter.

        Splits on these separators in order:
        1. Double newline (paragraph)
        2. Single newline
        3. Period + space (sentence)
        4. Space (word)
        5. Empty string (character)
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            is_separator_regex=False,
            add_start_index=True,
        )

        return splitter.split_documents(documents)

    def _semantic_chunk(
        self,
        documents: list[Document],
        max_chunk_size: int,
    ) -> list[Document]:
        """
        Split using semantic similarity between sentences.

        Groups adjacent sentences that are semantically similar,
        creating more coherent chunks. Falls back to recursive
        splitting if semantic chunking fails.
        """
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5",
                model_kwargs={"device": "cpu"},
            )

            splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95,
            )

            chunks = splitter.split_documents(documents)

            # Post-process: split any chunks that exceed max size
            final_chunks = []
            recursive_fallback = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=200,
            )

            for chunk in chunks:
                if len(chunk.page_content) > max_chunk_size:
                    sub_chunks = recursive_fallback.split_documents([chunk])
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)

            return final_chunks

        except ImportError:
            logger.warning("Semantic chunker not available, falling back to recursive")
            return self._recursive_chunk(documents, max_chunk_size, 200)
        except Exception as e:
            logger.error("Semantic chunking failed, falling back to recursive", error=str(e))
            return self._recursive_chunk(documents, max_chunk_size, 200)

    @staticmethod
    def _enrich_metadata(
        chunks: list[Document],
        original_documents: list[Document],
    ) -> list[Document]:
        """
        Enrich chunk metadata with positional and source information.

        Adds:
        - chunk_index: Position within the source document
        - total_chunks: Total chunks from the source document
        - doc_id: Unique identifier for deduplication
        """
        import hashlib

        # Group chunks by source
        source_chunks: dict[str, list[Document]] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            source_chunks.setdefault(source, []).append(chunk)

        # Enrich each chunk
        enriched = []
        for source, source_chunk_list in source_chunks.items():
            for idx, chunk in enumerate(source_chunk_list):
                # Generate deterministic ID for deduplication
                content_hash = hashlib.md5(
                    chunk.page_content.encode()
                ).hexdigest()[:12]

                enriched_chunk = Document(
                    page_content=chunk.page_content,
                    metadata={
                        **chunk.metadata,
                        "chunk_index": idx,
                        "total_chunks": len(source_chunk_list),
                        "doc_id": f"{source}_{content_hash}",
                        "chunk_size": len(chunk.page_content),
                    },
                )
                enriched.append(enriched_chunk)

        return enriched
