"""
Reranker Node — FlashRank cross-encoder reranking with noise filtering.

WHY: Initial retrieval optimizes for recall (find anything relevant). Reranking
optimizes for precision (keep only the best). Cross-encoder models like FlashRank
are dramatically more accurate than bi-encoder similarity because they see the
query and document together, enabling fine-grained relevance scoring.

ARCHITECTURE DECISION: FlashRank over larger cross-encoders (e.g., Cohere Rerank)
because:
1. Runs locally — no API calls, no latency, no cost per query
2. ~50ms for 20 documents — fast enough for real-time
3. Model size ~30MB — fits in any container
4. Accuracy competitive with API-based rerankers for most use cases

TRADE-OFF: FlashRank is less accurate than Cohere Rerank v3 or Voyage AI Rerank
on some benchmarks. For enterprise-critical applications where every percentage
point of precision matters, consider swapping to an API-based reranker.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from src.agent.state import AgentState
from src.config.logging_config import get_logger

logger = get_logger(__name__)


async def reranker_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Rerank retrieved documents using FlashRank cross-encoder.

    Reads: query, documents
    Writes: reranked_documents, metadata.retrieval (updated)

    Pipeline:
    1. Score each document against the query using cross-encoder
    2. Filter by score threshold (remove noise)
    3. Take top-N results
    4. Attach rerank scores to document metadata
    """
    start_time = time.perf_counter()
    query = state["query"]
    documents = state.get("documents", [])

    top_n = config.get("configurable", {}).get("rerank_top_n", 5)
    threshold = config.get("configurable", {}).get("rerank_threshold", 0.3)

    logger.info(
        "Reranker executing",
        input_docs=len(documents),
        top_n=top_n,
        threshold=threshold,
    )

    if not documents:
        logger.warning("No documents to rerank")
        return {
            "reranked_documents": [],
            "metadata": {
                **state.get("metadata", {}),
                "node_execution_order": state.get("metadata", {}).get(
                    "node_execution_order", []
                ) + ["reranker"],
            },
        }

    try:
        reranker = config["configurable"]["reranker"]

        # Prepare passages for reranking
        passages = [
            {
                "id": i,
                "text": doc.page_content,
                "meta": doc.metadata,
            }
            for i, doc in enumerate(documents)
        ]

        # Run reranking
        reranked_results = reranker.rerank(
            query=query,
            passages=passages,
            top_n=min(top_n * 2, len(passages)),  # Get more for threshold filtering
        )

        # Filter by threshold and take top-N
        reranked_documents: list[Document] = []
        for result in reranked_results:
            score = result.get("score", 0.0)
            if score < threshold:
                continue

            original_idx = result.get("id", 0)
            if original_idx < len(documents):
                doc = documents[original_idx]
                # Attach rerank score to metadata
                enriched_doc = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "rerank_score": round(score, 4),
                        "rerank_position": len(reranked_documents),
                    },
                )
                reranked_documents.append(enriched_doc)

            if len(reranked_documents) >= top_n:
                break

        latency_ms = (time.perf_counter() - start_time) * 1000
        filtered_count = len(documents) - len(reranked_documents)

        logger.info(
            "Reranker completed",
            input_docs=len(documents),
            output_docs=len(reranked_documents),
            filtered_out=filtered_count,
            latency_ms=round(latency_ms, 2),
        )

        # Update retrieval metadata
        existing_retrieval = state.get("metadata", {}).get("retrieval", {})
        return {
            "reranked_documents": reranked_documents,
            "metadata": {
                **state.get("metadata", {}),
                "retrieval": {
                    **existing_retrieval,
                    "total_after_rerank": len(reranked_documents),
                    "rerank_latency_ms": round(latency_ms, 2),
                },
                "node_execution_order": state.get("metadata", {}).get(
                    "node_execution_order", []
                ) + ["reranker"],
            },
        }

    except Exception as e:
        logger.error("Reranker failed, passing through documents unranked", error=str(e))
        # Graceful degradation: pass through original documents (top-N)
        return {
            "reranked_documents": documents[:top_n],
            "metadata": {
                **state.get("metadata", {}),
                "errors": state.get("metadata", {}).get("errors", [])
                + [f"reranker: {str(e)}"],
                "node_execution_order": state.get("metadata", {}).get(
                    "node_execution_order", []
                ) + ["reranker"],
            },
        }
