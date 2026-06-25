"""
Pydantic v2 schemas for API request/response validation.

WHY: Strict schema validation at the API boundary prevents invalid data from
reaching the core system. Pydantic v2 provides 5-50x faster validation than v1,
with better error messages and OpenAPI documentation generation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# === Common Models ===

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    incident_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthStatus(BaseModel):
    """Component health status."""
    status: str  # healthy | degraded | unhealthy
    components: dict[str, bool] = {}
    version: str = ""
    uptime_seconds: float = 0.0


# === Query Models ===

class QueryRequest(BaseModel):
    """POST /query request body."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User's question",
        examples=["What is our Q4 revenue?"],
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity. Auto-generated if not provided.",
    )
    user_id: Optional[str] = Field(
        "anonymous",
        description="User identifier",
    )
    filters: Optional[dict[str, Any]] = Field(
        None,
        description="Metadata filters for retrieval (e.g., {'doc_type': 'pdf'})",
    )
    stream: bool = Field(
        False,
        description="Enable streaming response via SSE",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query must not be empty or whitespace-only")
        return stripped


class CitationResponse(BaseModel):
    """Citation in the response."""
    source: str
    page: Optional[int] = None
    content_snippet: str = ""
    relevance_score: float = 0.0


class QueryResponse(BaseModel):
    """POST /query response body."""
    answer: str
    citations: list[CitationResponse] = []
    confidence: float = Field(ge=0.0, le=1.0)
    session_id: str
    intent: str = ""
    metadata: dict[str, Any] = {}


# === Ingest Models ===

class IngestResponse(BaseModel):
    """POST /ingest response body."""
    job_id: str
    status: str  # accepted | processing | completed | failed
    filename: str
    file_type: str
    message: str
    num_chunks: Optional[int] = None


class IngestStatusResponse(BaseModel):
    """Ingestion job status."""
    job_id: str
    status: str
    filename: str
    num_chunks: Optional[int] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# === Feedback Models ===

class FeedbackRequest(BaseModel):
    """POST /feedback request body."""
    session_id: str = Field(..., description="Session ID the feedback is for")
    message_id: Optional[str] = Field(None, description="Specific message ID")
    rating: int = Field(..., ge=1, le=5, description="Star rating (1-5)")
    comment: Optional[str] = Field(None, max_length=5000, description="Optional comment")


class FeedbackResponse(BaseModel):
    """POST /feedback response body."""
    id: str
    status: str = "recorded"
    message: str = "Thank you for your feedback"


# === Evaluation Models ===

class EvaluationRequest(BaseModel):
    """POST /evaluate request body."""
    dataset_name: Optional[str] = Field(
        None,
        description="Name of the golden dataset to evaluate against",
    )
    questions: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Inline Q&A pairs: [{'question': ..., 'ground_truth': ...}]",
    )
    metrics: list[str] = Field(
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        description="RAGAS metrics to compute",
    )

    @field_validator("questions", "dataset_name")
    @classmethod
    def validate_source(cls, v: Any, info: Any) -> Any:
        # At least one of questions or dataset_name must be provided
        return v


class EvaluationResponse(BaseModel):
    """POST /evaluate response body."""
    run_id: str
    status: str  # pending | running | completed | failed
    message: str


class EvaluationResultResponse(BaseModel):
    """Evaluation run results."""
    run_id: str
    status: str
    metrics: dict[str, float] = {}
    num_samples: int = 0
    duration_seconds: Optional[float] = None
    results: list[dict[str, Any]] = []


# === Metrics Models ===

class MetricsResponse(BaseModel):
    """GET /metrics response body."""
    total_queries: int = 0
    total_ingestions: int = 0
    avg_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    avg_confidence: float = 0.0
    error_rate: float = 0.0
    token_usage: dict[str, int] = {}
    active_sessions: int = 0
