"""
Responder Node — Context construction, answer generation, and citation extraction.

WHY: The responder is the user-facing output node. It assembles context from
reranked documents using Spotlighting (XML-delimited chunks), generates the
answer via LLM, and extracts citations. This is where hallucination reduction
happens through grounded generation.

ARCHITECTURE DECISION: Using Spotlighting (XML-delimited context) instead of
naive concatenation because:
1. Clear boundaries between source documents prevent cross-contamination
2. LLMs can attribute claims to specific chunks more accurately
3. Metadata (source, page) is directly visible to the model
4. Reduces hallucination by making the grounding boundary explicit
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agent.prompts.system_prompts import (
    CHITCHAT_RESPONSE,
    NO_CONTEXT_RESPONSE,
    RESPONDER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from src.agent.state import AgentState, Citation
from src.config.constants import Intent
from src.config.logging_config import get_logger

logger = get_logger(__name__)


async def responder_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Generate the final response with citations.

    Reads: query, intent, reranked_documents, chat_history
    Writes: response, citations, confidence, metadata.llm

    Pipeline:
    1. Check intent — route chitchat/clarification to templates
    2. Build XML-delimited context from reranked documents (Spotlighting)
    3. Call LLM with system prompt + context + query
    4. Extract citations from response
    5. Estimate confidence
    """
    start_time = time.perf_counter()
    query = state["query"]
    intent = state.get("intent", Intent.RAG)
    documents = state.get("reranked_documents", [])
    chat_history = state.get("chat_history", [])

    # Reranker fallback: if no documents survived reranking, but some were retrieved, use them
    is_rerank_fallback = False
    if not documents and state.get("documents"):
        documents = state.get("documents", [])[:3]
        is_rerank_fallback = True

    logger.info(
        "Responder executing",
        intent=intent,
        num_docs=len(documents),
        rerank_fallback=is_rerank_fallback,
    )

    # === Handle non-RAG intents ===
    if intent == Intent.CHITCHAT:
        return _build_response(state, CHITCHAT_RESPONSE, [], 1.0, start_time)

    if intent == Intent.OUT_OF_SCOPE:
        return _build_response(
            state,
            "I'm sorry, but that question falls outside my area of expertise. "
            "I can help you with questions about your uploaded documents.",
            [],
            1.0,
            start_time,
        )

    if intent == Intent.CLARIFICATION:
        return _build_response(
            state,
            "Could you please provide more details about what you're looking for? "
            "A more specific question will help me find the most relevant information.",
            [],
            1.0,
            start_time,
        )

    # === RAG Response Generation ===
    if not documents:
        return _build_response(state, NO_CONTEXT_RESPONSE, [], 0.0, start_time)

    try:
        gateway = config["configurable"]["gateway"]

        # Build Spotlighting context (XML-delimited chunks)
        context = _build_spotlight_context(documents)

        # Build conversation history for context
        history_str = ""
        if chat_history:
            recent = chat_history[-6:]  # Last 3 turns
            history_str = "\n\nPrevious conversation:\n" + "\n".join(
                f"{'User' if i % 2 == 0 else 'Assistant'}: {msg.content[:500]}"
                for i, msg in enumerate(recent)
            )

        # Construct the user message
        user_message = (
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            f"{history_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Generate a comprehensive, well-cited answer based on the context above."
        )

        messages = [
            SystemMessage(content=RESPONDER_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Call LLM via Portkey gateway
        llm_response = await gateway.ainvoke(
            messages,
            temperature=config.get("configurable", {}).get("temperature", 0.1),
            max_tokens=config.get("configurable", {}).get("max_tokens", 4096),
        )

        response_text = llm_response.content

        # Extract citations
        citations = _extract_citations(response_text, documents)

        # Estimate confidence
        confidence = _estimate_confidence(response_text, documents, citations)

        # Get token usage from response
        usage = getattr(llm_response, "usage_metadata", {}) or {}

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Responder completed",
            response_length=len(response_text),
            num_citations=len(citations),
            confidence=confidence,
            latency_ms=round(latency_ms, 2),
        )

        return {
            "response": response_text,
            "citations": citations,
            "confidence": confidence,
            "metadata": {
                **state.get("metadata", {}),
                "llm": {
                    "model": config.get("configurable", {}).get("model", "unknown"),
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "latency_ms": round(latency_ms, 2),
                    "cached": False,
                },
                "node_execution_order": state.get("metadata", {}).get(
                    "node_execution_order", []
                ) + ["responder"],
            },
        }

    except Exception as e:
        logger.error("Responder failed", error=str(e))
        if documents:
            chunks_text = []
            citations = []
            for i, doc in enumerate(documents[:3]):
                source_path = doc.metadata.get("source", "document")
                filename = source_path.rsplit("/", 1)[-1]
                page = doc.metadata.get("page")
                page_info = f", Page: {page}" if page else ""
                chunks_text.append(f"• **[Source: {filename}{page_info}]**:\n\"{doc.page_content.strip()}\"")

                citations.append(
                    Citation(
                        source=filename,
                        page=int(page) if page is not None else None,
                        content_snippet=doc.page_content[:200],
                        relevance_score=float(doc.metadata.get("rerank_score", 0.5)),
                    )
                )

            response_text = (
                "⚠️ **Local Heuristic Mode** (LLM API key not configured or failed):\n\n"
                "I retrieved the following highly relevant passages from your uploaded document to help answer your question:\n\n"
                + "\n\n".join(chunks_text)
            )

            return {
                "response": response_text,
                "citations": citations,
                "confidence": 0.5,
                "metadata": {
                    **state.get("metadata", {}),
                    "errors": state.get("metadata", {}).get("errors", [])
                    + [f"responder fallback: {str(e)}"],
                    "node_execution_order": state.get("metadata", {}).get(
                        "node_execution_order", []
                    ) + ["responder"],
                },
            }

        return {
            "response": (
                "I encountered an error while generating the response. "
                "Please try again or rephrase your question."
            ),
            "citations": [],
            "confidence": 0.0,
            "error": f"Responder error: {str(e)}",
            "metadata": {
                **state.get("metadata", {}),
                "errors": state.get("metadata", {}).get("errors", [])
                + [f"responder: {str(e)}"],
                "node_execution_order": state.get("metadata", {}).get(
                    "node_execution_order", []
                ) + ["responder"],
            },
        }


def _build_spotlight_context(documents: list) -> str:
    """
    Build XML-delimited Spotlighting context from documents.

    Each chunk is wrapped in XML tags with source metadata, making it easy
    for the LLM to attribute claims to specific sources.
    """
    chunks = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", f"document_{i}")
        page = doc.metadata.get("page", "N/A")
        score = doc.metadata.get("rerank_score", "N/A")

        chunk = (
            f'<chunk id="{i}" source="{source}" page="{page}" relevance="{score}">\n'
            f"{doc.page_content}\n"
            f"</chunk>"
        )
        chunks.append(chunk)

    return "\n\n".join(chunks)


def _extract_citations(response: str, documents: list) -> list[Citation]:
    """
    Extract source citations from the LLM response.

    Looks for [Source: X] or [Source: X, Page: Y] patterns.
    """
    citations: list[Citation] = []
    seen_sources: set[str] = set()

    # Pattern: [Source: filename] or [Source: filename, Page: N]
    pattern = r'\[Source:\s*([^\],]+?)(?:,\s*Page:\s*(\d+))?\]'
    matches = re.finditer(pattern, response)

    for match in matches:
        source = match.group(1).strip()
        page = int(match.group(2)) if match.group(2) else None

        if source in seen_sources:
            continue
        seen_sources.add(source)

        # Find matching document for content snippet
        snippet = ""
        relevance = 0.0
        for doc in documents:
            doc_source = doc.metadata.get("source", "")
            if source.lower() in doc_source.lower() or doc_source.lower() in source.lower():
                snippet = doc.page_content[:200]
                relevance = doc.metadata.get("rerank_score", 0.0)
                break

        citations.append(
            Citation(
                source=source,
                page=page,
                content_snippet=snippet,
                relevance_score=relevance,
            )
        )

    return citations


def _estimate_confidence(
    response: str, documents: list, citations: list[Citation]
) -> float:
    """
    Estimate response confidence based on evidence quality.

    Heuristic scoring:
    - Number of supporting documents (0.3 weight)
    - Average rerank score (0.3 weight)
    - Citation count (0.2 weight)
    - Response contains hedging language (0.2 weight)
    """
    if not documents:
        return 0.0

    # Document support score
    doc_score = min(len(documents) / 3.0, 1.0) * 0.3

    # Average rerank score
    rerank_scores = [
        doc.metadata.get("rerank_score", 0.5) for doc in documents
    ]
    avg_rerank = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0.0
    rerank_score = avg_rerank * 0.3

    # Citation coverage
    citation_score = min(len(citations) / 2.0, 1.0) * 0.2

    # Hedging language penalty
    hedging_phrases = [
        "i don't have enough information",
        "i'm not sure",
        "the context doesn't",
        "no relevant information",
        "cannot determine",
        "unclear from the documents",
    ]
    response_lower = response.lower()
    has_hedging = any(phrase in response_lower for phrase in hedging_phrases)
    hedging_score = 0.0 if has_hedging else 0.2

    confidence = doc_score + rerank_score + citation_score + hedging_score
    return round(min(confidence, 1.0), 2)


def _build_response(
    state: AgentState,
    response: str,
    citations: list[Citation],
    confidence: float,
    start_time: float,
) -> dict[str, Any]:
    """Build a standardized response dict for non-RAG intents."""
    latency_ms = (time.perf_counter() - start_time) * 1000
    return {
        "response": response,
        "citations": citations,
        "confidence": confidence,
        "metadata": {
            **state.get("metadata", {}),
            "node_execution_order": state.get("metadata", {}).get(
                "node_execution_order", []
            ) + ["responder"],
            "llm": {"latency_ms": round(latency_ms, 2), "cached": False},
        },
    }
