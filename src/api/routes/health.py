"""
Health, metrics, feedback, and evaluation routes.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request

from src.api.middleware.auth import verify_api_key
from src.api.schemas.common import (
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResultResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthStatus,
    MetricsResponse,
)
from src.config.logging_config import get_logger

logger = get_logger(__name__)

# === Health Route ===
health_router = APIRouter(tags=["Health"])


@health_router.get(
    "/health",
    response_model=HealthStatus,
    summary="System health check",
    description="Check the health of all system components.",
)
async def health_check(request: Request) -> HealthStatus:
    """Check health of all system components."""
    components: dict[str, bool] = {}

    # Check PostgreSQL
    try:
        memory = request.app.state.memory_store
        components["postgresql"] = await memory.health_check()
    except Exception:
        components["postgresql"] = False

    # Check Qdrant
    try:
        vectorstore = request.app.state.vectorstore
        components["qdrant"] = await vectorstore.health_check()
    except Exception:
        components["qdrant"] = False

    # Check embedding model
    try:
        embeddings = request.app.state.embedding_manager
        components["embeddings"] = await embeddings.health_check()
    except Exception:
        components["embeddings"] = False

    # Overall status
    all_healthy = all(components.values())
    any_healthy = any(components.values())

    if all_healthy:
        status = "healthy"
    elif any_healthy:
        status = "degraded"
    else:
        status = "unhealthy"

    uptime = time.time() - request.app.state.start_time

    return HealthStatus(
        status=status,
        components=components,
        version=request.app.state.settings.app_version,
        uptime_seconds=round(uptime, 2),
    )


# === Metrics Route ===
metrics_router = APIRouter(tags=["Metrics"])


@metrics_router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="System metrics",
    description="Get system metrics including query counts, latency, and token usage.",
)
async def get_metrics(
    request: Request,
    _api_key: str = Depends(verify_api_key),
) -> MetricsResponse:
    """Return system metrics."""
    collector = getattr(request.app.state, "metrics_collector", None)
    if collector:
        return MetricsResponse(**collector.get_summary())
    return MetricsResponse()


# === Feedback Route ===
feedback_router = APIRouter(tags=["Feedback"])


@feedback_router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback",
    description="Submit feedback on a response (rating and optional comment).",
)
async def submit_feedback(
    body: FeedbackRequest,
    request: Request,
    _api_key: str = Depends(verify_api_key),
) -> FeedbackResponse:
    """Record user feedback on a response."""
    feedback_id = str(uuid.uuid4())

    logger.info(
        "Feedback received",
        feedback_id=feedback_id,
        session_id=body.session_id,
        rating=body.rating,
    )

    # Store feedback
    try:
        from sqlalchemy import select
        from src.memory.models import Feedback

        memory = request.app.state.memory_store
        async with memory.session_factory() as db:
            feedback = Feedback(
                id=uuid.UUID(feedback_id),
                session_id=uuid.UUID(body.session_id),
                message_id=uuid.UUID(body.message_id) if body.message_id else None,
                rating=body.rating,
                comment=body.comment,
            )
            db.add(feedback)
            await db.commit()

    except Exception as e:
        logger.error("Failed to store feedback", error=str(e))

    return FeedbackResponse(
        id=feedback_id,
        status="recorded",
        message="Thank you for your feedback",
    )


# === Evaluation Route ===
evaluation_router = APIRouter(tags=["Evaluation"])


@evaluation_router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Trigger evaluation",
    description="Trigger a RAGAS evaluation run against a golden dataset.",
)
async def trigger_evaluation(
    body: EvaluationRequest,
    request: Request,
    _api_key: str = Depends(verify_api_key),
) -> EvaluationResponse:
    """Trigger a RAGAS evaluation run."""
    run_id = str(uuid.uuid4())

    logger.info(
        "Evaluation requested",
        run_id=run_id,
        dataset=body.dataset_name,
        metrics=body.metrics,
    )

    # Trigger async evaluation (non-blocking)
    evaluator = getattr(request.app.state, "evaluator", None)
    if evaluator:
        import asyncio
        asyncio.create_task(
            evaluator.run_evaluation(
                run_id=run_id,
                dataset_name=body.dataset_name,
                questions=body.questions,
                metrics=body.metrics,
            )
        )

    return EvaluationResponse(
        run_id=run_id,
        status="pending",
        message="Evaluation run started. Check status via the evaluation dashboard.",
    )


@evaluation_router.get(
    "/evaluate/runs",
    response_model=list[dict[str, Any]],
    summary="Get evaluation runs",
    description="Get all evaluation runs.",
)
async def get_evaluation_runs(
    request: Request,
    _api_key: str = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Return all evaluation runs."""
    memory = request.app.state.memory_store
    from sqlalchemy import select
    from src.memory.models import EvaluationRun

    async with memory.session_factory() as db:
        result = await db.execute(select(EvaluationRun).order_by(EvaluationRun.created_at.desc()))
        runs = result.scalars().all()
        return [
            {
                "run_id": str(run.id),
                "name": run.name,
                "dataset_name": run.dataset_name,
                "created_at": run.created_at.isoformat(),
                "status": run.status,
                "metrics": run.metrics or {},
                "num_samples": run.num_samples or 0,
                "duration_seconds": run.duration_seconds,
                "error": run.error,
            }
            for run in runs
        ]


@evaluation_router.get(
    "/evaluate/runs/{run_id}",
    response_model=EvaluationResultResponse,
    summary="Get evaluation run details",
    description="Get details and sample results for a specific evaluation run.",
)
async def get_evaluation_run_details(
    run_id: str,
    request: Request,
    _api_key: str = Depends(verify_api_key),
) -> EvaluationResultResponse:
    """Return specific evaluation run details and results."""
    memory = request.app.state.memory_store
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from src.memory.models import EvaluationRun

    async with memory.session_factory() as db:
        result = await db.execute(
            select(EvaluationRun)
            .where(EvaluationRun.id == uuid.UUID(run_id))
            .options(selectinload(EvaluationRun.results))
        )
        run = result.scalar_one_or_none()
        if not run:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Evaluation run not found")

        return EvaluationResultResponse(
            run_id=str(run.id),
            status=run.status,
            metrics=run.metrics or {},
            num_samples=run.num_samples or 0,
            duration_seconds=run.duration_seconds,
            results=[
                {
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "generated_answer": r.generated_answer,
                    "contexts": r.contexts,
                    "metrics": r.metrics,
                }
                for r in run.results
            ],
        )
