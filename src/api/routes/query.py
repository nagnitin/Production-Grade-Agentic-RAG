"""
Query route — POST /query endpoint.

The main endpoint for processing user queries through the agentic RAG pipeline.
Supports both standard JSON responses and streaming via SSE.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request

from src.api.middleware.auth import verify_api_key
from src.api.middleware.rate_limiter import rate_limit_dependency
from src.api.schemas.common import CitationResponse, ErrorResponse, QueryRequest, QueryResponse
from src.config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Process a user query through the RAG pipeline",
    description=(
        "Sends a user query through the agentic RAG pipeline: "
        "Planner → Retriever → Reranker → Responder → Memory. "
        "Returns an answer with citations and confidence score."
    ),
)
async def query_endpoint(
    body: QueryRequest,
    request: Request,
    _api_key: str = Depends(verify_api_key),
    _rate_limit: None = Depends(rate_limit_dependency),
) -> QueryResponse:
    """Process a user query through the agentic RAG pipeline."""
    start_time = time.perf_counter()
    session_id = body.session_id or str(uuid.uuid4())
    user_id = body.user_id or "anonymous"

    logger.info(
        "Query received",
        query=body.query[:100],
        session_id=session_id,
        user_id=user_id,
    )

    # Get dependencies from app state
    app = request.app
    graph = app.state.agent_graph
    gateway = app.state.gateway
    vectorstore = app.state.vectorstore
    reranker = app.state.reranker
    memory_store = app.state.memory_store
    semantic_cache = app.state.semantic_cache
    guardrails_manager = getattr(app.state, "guardrails_manager", None)

    # === Check semantic cache ===
    if semantic_cache:
        cached = await semantic_cache.get(body.query)
        if cached:
            logger.info("Returning cached response", session_id=session_id)
            metrics_collector = getattr(app.state, "metrics_collector", None)
            if metrics_collector:
                metrics_collector.record_query(
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    confidence=cached.get("confidence", 0.0),
                    cache_hit=True,
                )
                metrics_collector.record_session(session_id)
            return QueryResponse(
                answer=cached["response"],
                citations=[
                    CitationResponse(**c) for c in cached.get("citations", [])
                ],
                confidence=cached.get("confidence", 0.0),
                session_id=session_id,
                intent="cached",
                metadata={"cached": True, "cache_similarity": cached.get("cache_similarity", 0.0)},
            )

    # === Run input guardrails ===
    if guardrails_manager:
        guard_result = await guardrails_manager.check_input(body.query)
        if not guard_result["allowed"]:
            logger.warning(
                "Query blocked by guardrails",
                reason=guard_result.get("reason", "unknown"),
            )
            metrics_collector = getattr(app.state, "metrics_collector", None)
            if metrics_collector:
                metrics_collector.record_query(
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    confidence=0.0,
                    cache_hit=False,
                    error=True,
                )
                metrics_collector.record_session(session_id)
            return QueryResponse(
                answer=guard_result.get(
                    "message",
                    "I'm unable to process this request. Please rephrase your question.",
                ),
                citations=[],
                confidence=1.0,
                session_id=session_id,
                intent="blocked",
                metadata={"guardrails_blocked": True, "reason": guard_result.get("reason")},
            )

    # === Run the agent graph ===
    from src.agent.graph import run_agent

    result = await run_agent(
        graph=graph,
        query=body.query,
        session_id=session_id,
        user_id=user_id,
        filters=body.filters,
        config={
            "gateway": gateway,
            "vectorstore": vectorstore,
            "reranker": reranker,
            "memory_store": memory_store,
        },
    )

    # === Run output guardrails ===
    response_text = result.get("response", "")
    if guardrails_manager:
        output_result = await guardrails_manager.check_output(response_text)
        if not output_result["allowed"]:
            response_text = output_result.get(
                "sanitized",
                "The response was filtered for safety. Please try a different question.",
            )

    # === Cache the response ===
    if semantic_cache and result.get("confidence", 0) > 0.5:
        await semantic_cache.set(
            query=body.query,
            response=response_text,
            citations=result.get("citations", []),
            confidence=result.get("confidence", 0.0),
        )

    # === Build response ===
    citations = [
        CitationResponse(
            source=c.get("source", ""),
            page=c.get("page"),
            content_snippet=c.get("content_snippet", ""),
            relevance_score=c.get("relevance_score", 0.0),
        )
        for c in result.get("citations", [])
    ]

    # Record metrics
    metrics_collector = getattr(app.state, "metrics_collector", None)
    if metrics_collector:
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics_collector.record_query(
            latency_ms=latency_ms,
            confidence=result.get("confidence", 0.0),
            cache_hit=False,
        )
        metrics_collector.record_session(session_id)
        
        metadata = result.get("metadata", {})
        llm_meta = metadata.get("llm", {}) if isinstance(metadata, dict) else {}
        if isinstance(llm_meta, dict):
            prompt_tokens = llm_meta.get("prompt_tokens", 0)
            completion_tokens = llm_meta.get("completion_tokens", 0)
            metrics_collector.record_tokens(prompt_tokens, completion_tokens)

    return QueryResponse(
        answer=response_text,
        citations=citations,
        confidence=result.get("confidence", 0.0),
        session_id=session_id,
        intent=result.get("intent", ""),
        metadata=result.get("metadata", {}),
    )
