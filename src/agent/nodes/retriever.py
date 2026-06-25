"""
Retriever Node — Vector retrieval from Qdrant with hybrid search support.

WHY: Retrieval quality is the single most important factor in RAG. A dedicated
retriever node handles strategy selection (dense/sparse/hybrid), metadata
filtering, and multi-query retrieval for decomposed queries.

ARCHITECTURE DECISION: Pre-rerank retrieval fetches top-K (default: 20) documents.
Over-fetching intentionally — the reranker will filter down to top-N (default: 5).
This two-stage approach (retrieve broadly → rerank precisely) consistently
outperforms single-stage retrieval in production benchmarks.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from src.agent.state import AgentState
from src.config.constants import RetrievalStrategy
from src.config.logging_config import get_logger

logger = get_logger(__name__)


async def retriever_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Retrieve relevant documents from Qdrant.

    Reads: query, retrieval_strategy, query_decomposition, filters
    Writes: documents, metadata.retrieval

    Supports three retrieval strategies:
    - Dense: Semantic similarity search using embeddings
    - Sparse: Keyword-based search using BM25-like scoring
    - Hybrid: Combined dense + sparse with Reciprocal Rank Fusion
    """
    start_time = time.perf_counter()
    query = state["query"]
    strategy = state.get("retrieval_strategy", RetrievalStrategy.HYBRID)
    decomposed_queries = state.get("query_decomposition", [query])
    filters = state.get("filters", {})
    top_k = config.get("configurable", {}).get("top_k", 20)

    logger.info(
        "Retriever executing",
        strategy=strategy,
        num_sub_queries=len(decomposed_queries),
        top_k=top_k,
    )

    try:
        vectorstore = config["configurable"]["vectorstore"]
        all_documents: list[Document] = []
        seen_ids: set[str] = set()

        for sub_query in decomposed_queries:
            if strategy == RetrievalStrategy.DENSE:
                docs = await vectorstore.dense_search(
                    query=sub_query,
                    top_k=top_k,
                    filters=filters,
                )
            elif strategy == RetrievalStrategy.SPARSE:
                docs = await vectorstore.sparse_search(
                    query=sub_query,
                    top_k=top_k,
                    filters=filters,
                )
            elif strategy == RetrievalStrategy.HYBRID:
                docs = await vectorstore.hybrid_search(
                    query=sub_query,
                    top_k=top_k,
                    filters=filters,
                )
            else:
                # Fallback to hybrid
                docs = await vectorstore.hybrid_search(
                    query=sub_query,
                    top_k=top_k,
                    filters=filters,
                )

            # Deduplicate across sub-queries
            for doc in docs:
                doc_id = doc.metadata.get("doc_id", doc.page_content[:100])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_documents.append(doc)

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Retriever completed",
            total_docs=len(all_documents),
            strategy=strategy,
            latency_ms=round(latency_ms, 2),
        )

        return {
            "documents": all_documents,
            "metadata": {
                **state.get("metadata", {}),
                "retrieval": {
                    "strategy": strategy,
                    "total_retrieved": len(all_documents),
                    "retrieval_latency_ms": round(latency_ms, 2),
                },
                "node_execution_order": state.get("metadata", {}).get(
                    "node_execution_order", []
                ) + ["retriever"],
            },
        }

    except Exception as e:
        logger.error("Retriever failed", error=str(e), strategy=strategy)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "documents": [],
            "error": f"Retrieval error: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1,
            "metadata": {
                **state.get("metadata", {}),
                "retrieval": {
                    "strategy": strategy,
                    "total_retrieved": 0,
                    "retrieval_latency_ms": round(latency_ms, 2),
                },
                "errors": state.get("metadata", {}).get("errors", [])
                + [f"retriever: {str(e)}"],
                "node_execution_order": state.get("metadata", {}).get(
                    "node_execution_order", []
                ) + ["retriever"],
            },
        }
