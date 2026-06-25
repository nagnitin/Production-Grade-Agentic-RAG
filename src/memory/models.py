"""
SQLAlchemy models for conversation memory.

WHY: Structured persistence of conversations enables:
1. Multi-session support — users can resume conversations
2. Analytics — query patterns, popular topics, feedback correlation
3. Debugging — full conversation replay for issue investigation
4. Compliance — audit trail of all interactions
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class Session(Base):
    """Conversation session."""

    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    title = Column(String(500), nullable=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    metadata_ = Column("metadata", JSON, default=dict)
    is_active = Column(Integer, default=1)  # 1=active, 0=archived

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_sessions_user_active", "user_id", "is_active"),
        Index("ix_sessions_updated", "updated_at"),
    )


class Message(Base):
    """Individual conversation message."""

    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String(20), nullable=False)  # "human" or "ai"
    content = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    metadata_ = Column("metadata", JSON, default=dict)

    # AI response metadata
    confidence = Column(Float, nullable=True)
    citations_count = Column(Integer, nullable=True)
    intent = Column(String(50), nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)

    session = relationship("Session", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
    )


class Feedback(Base):
    """User feedback on responses."""

    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    message_id = Column(UUID(as_uuid=True), nullable=True)
    rating = Column(Integer, nullable=False)  # 1-5 star rating
    comment = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    metadata_ = Column("metadata", JSON, default=dict)

    session = relationship("Session", back_populates="feedback")

    __table_args__ = (
        Index("ix_feedback_session", "session_id"),
    )


class EvaluationRun(Base):
    """RAGAS evaluation run results."""

    __tablename__ = "evaluation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    dataset_name = Column(String(255), nullable=False)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    metrics = Column(JSON, nullable=False, default=dict)
    config = Column(JSON, nullable=True)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    error = Column(Text, nullable=True)
    num_samples = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    results = relationship(
        "EvaluationResult", back_populates="run", cascade="all, delete-orphan"
    )


class EvaluationResult(Base):
    """Individual evaluation result per sample."""

    __tablename__ = "evaluation_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=True)
    generated_answer = Column(Text, nullable=True)
    contexts = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=False, default=dict)

    run = relationship("EvaluationRun", back_populates="results")


class DocumentRecord(Base):
    """Record of ingested documents."""

    __tablename__ = "document_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(500), nullable=False)
    source_path = Column(String(1000), nullable=True)
    file_type = Column(String(50), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    num_chunks = Column(Integer, nullable=True)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error = Column(Text, nullable=True)
    content_hash = Column(String(64), nullable=True, unique=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    processed_at = Column(DateTime(timezone=True), nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)

    __table_args__ = (
        Index("ix_document_records_hash", "content_hash"),
        Index("ix_document_records_status", "status"),
    )
