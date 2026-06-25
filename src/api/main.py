"""
FastAPI application factory.

WHY: The app factory pattern enables:
1. Testing — create isolated app instances with mock dependencies
2. Configuration — different configs for dev/staging/prod
3. Lifecycle management — clean startup/shutdown of resources

ARCHITECTURE DECISION: Using FastAPI's lifespan context manager for resource
lifecycle. All heavy resources (DB pools, ML models, Qdrant clients) are
initialized on startup and cleaned up on shutdown. This ensures no resource
leaks and proper connection cleanup.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.error_handler import register_exception_handlers
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.api.routes.health import (
    evaluation_router,
    feedback_router,
    health_router,
    metrics_router,
)
from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.config.logging_config import get_logger, setup_logging
from src.config.settings import get_settings

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan — initialize and cleanup resources.

    Startup:
    - Load configuration
    - Initialize logging
    - Connect to PostgreSQL
    - Connect to Qdrant
    - Load embedding model
    - Load reranker model
    - Build LangGraph agent
    - Initialize guardrails

    Shutdown:
    - Close database connections
    - Close Qdrant client
    - Cleanup resources
    """
    settings = get_settings()
    setup_logging(settings)

    logger.info(
        "Starting Agentic RAG Platform",
        version=settings.app_version,
        environment=settings.app_env.value,
    )

    app.state.settings = settings
    app.state.start_time = time.time()

    # === Initialize Metrics ===
    from src.observability.metrics import MetricsCollector
    metrics_collector = MetricsCollector(settings)
    app.state.metrics_collector = metrics_collector
    logger.info("Metrics collector initialized")

    # === Initialize PostgreSQL Memory ===
    from src.memory.postgres_memory import PostgresMemory

    memory_store = PostgresMemory(settings)
    await memory_store.initialize()
    app.state.memory_store = memory_store
    logger.info("PostgreSQL memory initialized")

    # === Initialize Embedding Manager ===
    from src.retrieval.embeddings import EmbeddingManager

    embedding_manager = EmbeddingManager(settings)
    app.state.embedding_manager = embedding_manager
    logger.info("Embedding manager initialized")

    # === Initialize Qdrant ===
    from src.retrieval.vectorstore import QdrantRetriever

    vectorstore = QdrantRetriever(settings, embedding_manager)
    await vectorstore.initialize()
    app.state.vectorstore = vectorstore
    logger.info("Qdrant vectorstore initialized")

    # === Initialize Reranker ===
    from src.retrieval.reranker import FlashRankReranker

    reranker = FlashRankReranker(settings)
    app.state.reranker = reranker
    logger.info("FlashRank reranker initialized")

    # === Initialize Portkey Gateway ===
    from src.gateway.portkey_client import PortkeyGateway

    gateway = PortkeyGateway(settings)
    app.state.gateway = gateway
    logger.info("Portkey gateway initialized")

    # === Initialize Semantic Cache ===
    from src.gateway.cache import SemanticCache

    semantic_cache = SemanticCache(settings, embedding_manager)
    await semantic_cache.initialize()
    app.state.semantic_cache = semantic_cache
    logger.info("Semantic cache initialized")

    # === Initialize Guardrails ===
    try:
        from src.guardrails.rails_manager import GuardrailsManager

        if settings.guardrails.enabled:
            guardrails_manager = GuardrailsManager(settings)
            await guardrails_manager.initialize()
            app.state.guardrails_manager = guardrails_manager
            logger.info("NeMo Guardrails initialized")
        else:
            app.state.guardrails_manager = None
    except Exception as e:
        logger.warning("Guardrails initialization failed, continuing without", error=str(e))
        app.state.guardrails_manager = None

    # === Initialize Ingestion Pipeline ===
    from src.ingestion.pipeline import IngestionPipeline

    ingestion_pipeline = IngestionPipeline(settings, vectorstore, embedding_manager)
    app.state.ingestion_pipeline = ingestion_pipeline
    logger.info("Ingestion pipeline initialized")

    # === Build LangGraph Agent ===
    from src.agent.graph import build_agent_graph

    agent_graph = build_agent_graph()
    app.state.agent_graph = agent_graph
    logger.info("Agent graph compiled")

    # === Initialize Ragas Evaluator ===
    try:
        from src.evaluation.ragas_evaluator import RagasEvaluator
        evaluator = RagasEvaluator(
            settings=settings,
            graph=agent_graph,
            memory_store=memory_store,
            graph_config={
                "gateway": gateway,
                "vectorstore": vectorstore,
                "reranker": reranker,
                "memory_store": memory_store,
            }
        )
        app.state.evaluator = evaluator
        logger.info("Ragas evaluator initialized")
    except Exception as e:
        logger.warning("Ragas evaluator initialization failed", error=str(e))
        app.state.evaluator = None

    # === Initialize Observability ===
    try:
        from src.observability.langsmith_tracer import setup_langsmith

        setup_langsmith(settings)
    except Exception as e:
        logger.warning("LangSmith setup failed", error=str(e))

    logger.info("All systems initialized — ready to serve")

    yield

    # === Shutdown ===
    logger.info("Shutting down...")

    await vectorstore.close()
    await memory_store.close()
    if semantic_cache:
        await semantic_cache.close()

    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This is the entry point for uvicorn:
        uvicorn src.api.main:create_app --factory
    """
    settings = get_settings()

    app = FastAPI(
        title="Agentic RAG Platform",
        description=(
            "Production-grade enterprise AI system with multi-document RAG, "
            "agentic planning, semantic reranking, and conversation memory."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
    )

    # === Middleware Stack (order matters: last added = first executed) ===

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging (wraps everything)
    app.add_middleware(LoggingMiddleware)

    # === Exception Handlers ===
    register_exception_handlers(app)

    # === Routes ===
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(health_router)
    app.include_router(metrics_router, prefix="/api/v1")
    app.include_router(feedback_router, prefix="/api/v1")
    app.include_router(evaluation_router, prefix="/api/v1")

    return app
