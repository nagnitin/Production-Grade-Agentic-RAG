"""
Conditional edge functions for LangGraph routing.

WHY: LangGraph conditional edges determine which node executes next based on
the current state. This enables dynamic routing (skip retrieval for chitchat),
retry logic, and confidence-based re-processing.

These functions are pure — they read state and return a node name string.
No side effects, no mutations.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.config.constants import Intent, NodeName
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def route_after_planner(state: AgentState) -> str:
    """
    Route after the planner node based on classified intent.

    - RAG queries → Retriever (full pipeline)
    - Chitchat/Clarification/Out-of-scope → Responder (skip retrieval)
    - Error state → Error handler
    """
    intent = state.get("intent", Intent.RAG)

    if state.get("error") and state.get("retry_count", 0) >= 3:
        logger.warning("Max retries exceeded, routing to responder with error")
        return NodeName.RESPONDER

    if intent == Intent.RAG:
        logger.info("Routing to retriever", intent=intent)
        return NodeName.RETRIEVER

    # Chitchat, clarification, out_of_scope — skip retrieval
    logger.info("Skipping retrieval, routing to responder", intent=intent)
    return NodeName.RESPONDER


def route_after_retriever(state: AgentState) -> str:
    """
    Route after retrieval.

    - Documents found → Reranker
    - No documents + no error → Responder (will handle empty context)
    - Error → Check if retry is warranted
    """
    documents = state.get("documents", [])
    error = state.get("error")

    if documents:
        logger.info("Documents retrieved, routing to reranker", count=len(documents))
        return NodeName.RERANKER

    if error and state.get("retry_count", 0) < 3:
        logger.warning("Retrieval error, attempting retry", error=error)
        return NodeName.RETRIEVER  # Retry

    logger.info("No documents found, routing to responder")
    return NodeName.RESPONDER


def route_after_reranker(state: AgentState) -> str:
    """
    Route after reranking. Always goes to responder.

    Future extension point: could route to a web search fallback
    if reranked documents are below a quality threshold.
    """
    reranked = state.get("reranked_documents", [])

    if not reranked:
        logger.warning("No documents survived reranking, routing to responder")

    return NodeName.RESPONDER


def route_after_responder(state: AgentState) -> str:
    """
    Route after response generation.

    Always goes to memory to persist the conversation turn.
    Future extension: could route to a Self-RAG reflect node
    for confidence checking and iterative refinement.
    """
    confidence = state.get("confidence", 1.0)

    if confidence < 0.3:
        logger.warning(
            "Low confidence response",
            confidence=confidence,
            response_preview=state.get("response", "")[:100],
        )

    return NodeName.MEMORY


def should_end(state: AgentState) -> str:
    """
    Determine if the graph should end after memory.

    This is the terminal edge — always returns END.
    """
    return "__end__"
