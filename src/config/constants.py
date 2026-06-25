"""
System-wide constants.

WHY: Centralizes magic strings and values that are used across multiple modules.
Changing a constant here propagates everywhere, reducing drift and errors.
"""

from __future__ import annotations

# === Collection Schemas ===
QDRANT_DOCUMENT_COLLECTION = "documents"
QDRANT_CACHE_COLLECTION = "semantic_cache"

# === Embedding ===
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_EMBEDDING_DIMENSION = 768

# === Chunking ===
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_CHUNK_SIZE = 4000
MIN_CHUNK_SIZE = 100

# === Retrieval ===
DEFAULT_TOP_K = 20
DEFAULT_RERANK_TOP_N = 5
DEFAULT_RERANK_THRESHOLD = 0.3
HYBRID_DENSE_WEIGHT = 0.7
HYBRID_SPARSE_WEIGHT = 0.3

# === LLM ===
PRIMARY_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
FALLBACK_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_CONTEXT_TOKENS = 8192
MAX_RESPONSE_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.1

# === Memory ===
MAX_CONVERSATION_HISTORY = 10
SUMMARIZE_THRESHOLD = 20
SESSION_TTL_HOURS = 24

# === Rate Limiting ===
DEFAULT_RATE_LIMIT = 100
DEFAULT_RATE_WINDOW = 60

# === File Processing ===
SUPPORTED_FILE_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/html": ".html",
    "text/plain": ".txt",
}

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".txt"}

MAX_FILE_SIZE_MB = 100
MAX_FILES_PER_REQUEST = 10

# === API ===
API_V1_PREFIX = "/api/v1"
HEALTH_ENDPOINT = "/health"
METRICS_ENDPOINT = "/metrics"

# === Cache ===
SEMANTIC_CACHE_THRESHOLD = 0.95
CACHE_TTL_SECONDS = 3600

# === Guardrails ===
MAX_INPUT_LENGTH = 10000
MAX_OUTPUT_LENGTH = 20000

# === Evaluation ===
RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

# === Observability ===
TRACE_SAMPLE_RATE = 1.0  # Sample 100% in dev, reduce in prod

# === Agent Intents ===
class Intent:
    """Agent intent classification labels."""
    RAG = "rag"
    CHITCHAT = "chitchat"
    CLARIFICATION = "clarification"
    OUT_OF_SCOPE = "out_of_scope"


# === Retrieval Strategies ===
class RetrievalStrategy:
    """Retrieval strategy labels."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


# === Node Names ===
class NodeName:
    """LangGraph node identifiers."""
    PLANNER = "planner"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    RESPONDER = "responder"
    MEMORY = "memory"
    GUARDRAILS_INPUT = "guardrails_input"
    GUARDRAILS_OUTPUT = "guardrails_output"
    ERROR_HANDLER = "error_handler"
