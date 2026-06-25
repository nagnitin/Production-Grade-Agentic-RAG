"""
PostgreSQL conversation memory with async support.

WHY: Persistent conversation memory enables multi-turn dialogue, session
resumption, and conversation analytics. Using async SQLAlchemy for non-blocking
database operations in the FastAPI async context.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config.logging_config import get_logger
from src.config.settings import Settings
from src.memory.models import Base, Message, Session

logger = get_logger(__name__)


class PostgresMemory:
    """
    Async PostgreSQL-backed conversation memory.

    Provides:
    - Message persistence (human + AI messages)
    - Session management
    - History retrieval with configurable limits
    - Message counting for summarization triggers
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def initialize(self) -> None:
        """Create engine, session factory, and ensure tables exist."""
        self._engine = create_async_engine(
            self.settings.postgres.async_url,
            pool_size=self.settings.postgres.pool_size,
            max_overflow=self.settings.postgres.max_overflow,
            pool_pre_ping=True,
            echo=self.settings.debug,
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create tables if they don't exist
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("PostgreSQL memory initialized")

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            raise RuntimeError("PostgresMemory not initialized. Call initialize() first.")
        return self._session_factory

    def _parse_uuid(self, val: str) -> uuid.UUID:
        try:
            return uuid.UUID(val)
        except ValueError:
            return uuid.uuid5(uuid.NAMESPACE_DNS, val)

    async def ensure_session(
        self,
        session_id: str,
        user_id: str = "anonymous",
    ) -> str:
        """Ensure a session exists, creating one if needed."""
        parsed_id = self._parse_uuid(session_id)
        async with self.session_factory() as db:
            result = await db.execute(
                select(Session).where(Session.id == parsed_id)
            )
            session = result.scalar_one_or_none()

            if not session:
                session = Session(
                    id=parsed_id,
                    user_id=user_id,
                    title="New Conversation",
                )
                db.add(session)
                await db.commit()
                logger.info("Created new session", session_id=session_id)

            return str(session.id)

    async def add_messages(
        self,
        session_id: str,
        messages: list[BaseMessage],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save messages to the database."""
        metadata = metadata or {}
        parsed_id = self._parse_uuid(session_id)

        async with self.session_factory() as db:
            # Ensure session exists
            await self.ensure_session(
                session_id,
                user_id=metadata.get("user_id", "anonymous"),
            )

            for msg in messages:
                role = "human" if isinstance(msg, HumanMessage) else "ai"

                db_message = Message(
                    session_id=parsed_id,
                    role=role,
                    content=msg.content,
                    confidence=metadata.get("confidence") if role == "ai" else None,
                    citations_count=metadata.get("citations_count") if role == "ai" else None,
                    intent=metadata.get("intent") if role == "ai" else None,
                    metadata_=metadata,
                )
                db.add(db_message)

            # Update session timestamp
            result = await db.execute(
                select(Session).where(Session.id == parsed_id)
            )
            session = result.scalar_one_or_none()
            if session:
                session.updated_at = datetime.now(timezone.utc)

            await db.commit()

    async def get_messages(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[BaseMessage]:
        """Retrieve recent messages for a session."""
        parsed_id = self._parse_uuid(session_id)
        async with self.session_factory() as db:
            result = await db.execute(
                select(Message)
                .where(Message.session_id == parsed_id)
                .order_by(desc(Message.created_at))
                .limit(limit)
            )
            db_messages = result.scalars().all()

            # Reverse to chronological order
            db_messages = list(reversed(db_messages))

            messages: list[BaseMessage] = []
            for msg in db_messages:
                if msg.role == "human":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))

            return messages

    async def get_message_count(self, session_id: str) -> int:
        """Count total messages in a session."""
        parsed_id = self._parse_uuid(session_id)
        async with self.session_factory() as db:
            result = await db.execute(
                select(func.count(Message.id)).where(
                    Message.session_id == parsed_id
                )
            )
            return result.scalar() or 0

    async def get_sessions(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List sessions for a user."""
        async with self.session_factory() as db:
            result = await db.execute(
                select(Session)
                .where(Session.user_id == user_id, Session.is_active == 1)
                .order_by(desc(Session.updated_at))
                .limit(limit)
            )
            sessions = result.scalars().all()

            return [
                {
                    "id": str(s.id),
                    "title": s.title,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                }
                for s in sessions
            ]

    async def clear_session(self, session_id: str) -> None:
        """Delete all messages in a session."""
        parsed_id = self._parse_uuid(session_id)
        async with self.session_factory() as db:
            result = await db.execute(
                select(Message).where(Message.session_id == parsed_id)
            )
            messages = result.scalars().all()
            for msg in messages:
                await db.delete(msg)
            await db.commit()

        logger.info("Cleared session", session_id=session_id)

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.session_factory() as db:
                await db.execute(select(func.count(Session.id)))
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close the database engine."""
        if self._engine:
            await self._engine.dispose()
            logger.info("PostgreSQL connection closed")
