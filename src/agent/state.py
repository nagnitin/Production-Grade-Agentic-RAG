"""
Typed agent state for LangGraph.

WHY: LangGraph requires a typed state that flows through all nodes. Using TypedDict
provides compile-time type checking while maintaining compatibility with LangGraph's
state management. Each node reads from and writes to specific state fields.

ARCHITECTURE DECISION: Using TypedDict over Pydantic BaseModel because LangGraph's
StateGraph requires dict-like state. TypedDict gives us the best of both worlds:
type annotations for IDE support and runtime compatibility with LangGraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class Citation(TypedDict):
    """A source citation for a generated response."""
    source: str
    page: Optional[int]
    content_snippet: str
    relevance_score: float


class RetrievalMetadata(TypedDict, total=False):
    """Metadata from the retrieval process."""
    strategy: str
    total_retrieved: int
    total_after_rerank: int
    retrieval_latency_ms: float
    rerank_latency_ms: float


class LLMMetadata(TypedDict, total=False):
    """Metadata from the LLM call."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cached: bool


class ExecutionMetadata(TypedDict, total=False):
    """Full execution metadata across all nodes."""
    retrieval: RetrievalMetadata
    llm: LLMMetadata
    total_latency_ms: float
    node_execution_order: list[str]
    guardrails_input_passed: bool
    guardrails_output_passed: bool
    errors: list[str]


class AgentState(TypedDict, total=False):
    """
    Typed state that flows through the LangGraph agent.

    Each field is read/written by specific nodes:
    - Planner: reads query, chat_history → writes intent, retrieval_strategy
    - Retriever: reads query, retrieval_strategy → writes documents
    - Reranker: reads documents → writes reranked_documents
    - Responder: reads query, reranked_documents, chat_history → writes response, citations
    - Memory: reads/writes chat_history, session_id

    The `messages` field uses LangGraph's `add_messages` reducer to automatically
    append messages rather than overwrite them.
    """

    # === Input ===
    query: str
    session_id: str
    user_id: str
    filters: dict[str, Any]

    # === Messages (with LangGraph reducer) ===
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # === Chat History (for memory retrieval) ===
    chat_history: list[BaseMessage]

    # === Planner Output ===
    intent: str
    retrieval_strategy: str
    query_decomposition: list[str]

    # === Retriever Output ===
    documents: list[Document]

    # === Reranker Output ===
    reranked_documents: list[Document]

    # === Responder Output ===
    response: str
    citations: list[Citation]
    confidence: float

    # === Guardrails ===
    guardrails_input_result: Optional[dict[str, Any]]
    guardrails_output_result: Optional[dict[str, Any]]
    guardrails_blocked: bool

    # === Execution Metadata ===
    metadata: ExecutionMetadata

    # === Error Handling ===
    error: Optional[str]
    retry_count: int
    should_retry: bool


@dataclass
class GraphConfig:
    """
    Configuration passed to the graph at runtime.

    This is separate from state — it contains configuration that doesn't change
    during execution (like feature flags and thresholds).
    """
    # Retrieval
    top_k: int = 20
    rerank_top_n: int = 5
    rerank_threshold: float = 0.3

    # LLM
    temperature: float = 0.1
    max_tokens: int = 4096

    # Memory
    max_history: int = 10

    # Guardrails
    guardrails_enabled: bool = True

    # Cache
    cache_enabled: bool = True
    cache_threshold: float = 0.95

    # Feature flags
    use_hybrid_search: bool = True
    use_query_decomposition: bool = False
    enable_human_in_the_loop: bool = False

    # Retry
    max_retries: int = 3

    # Metadata
    configurable: dict[str, Any] = field(default_factory=dict)
