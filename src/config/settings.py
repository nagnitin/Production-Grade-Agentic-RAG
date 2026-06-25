"""
Centralized configuration using Pydantic BaseSettings.

WHY: Type-safe configuration that validates on startup. Eliminates the class
of bugs caused by missing or malformed config values at runtime. Environment
variables are loaded automatically, with .env file support for local development.

ARCHITECTURE DECISION: Using nested Pydantic models for grouped config provides:
- IDE autocompletion and type checking
- Automatic validation with clear error messages
- Environment variable prefix isolation (e.g., POSTGRES_HOST, QDRANT_HOST)
- Secret file support for production (e.g., Docker secrets)
"""

from __future__ import annotations

import functools
from enum import Enum
from typing import Optional

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class PostgresSettings(BaseSettings):
    """PostgreSQL database configuration."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    db: str = "agentic_rag"
    user: str = "rag_user"
    password: SecretStr = SecretStr("changeme")
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def async_url(self) -> str:
        """Build async database URL."""
        pwd = self.password.get_secret_value()
        return f"postgresql+asyncpg://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"

    @property
    def sync_url(self) -> str:
        """Build sync database URL (for Alembic)."""
        pwd = self.password.get_secret_value()
        return f"postgresql+psycopg://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"

    @property
    def dsn(self) -> str:
        """Build psycopg DSN for LangGraph checkpoint saver."""
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"


class QdrantSettings(BaseSettings):
    """Qdrant vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[SecretStr] = None
    url: Optional[str] = None
    collection_name: str = "documents"
    collection_cache: str = "semantic_cache"
    prefer_grpc: bool = False
    timeout: int = 30

    @property
    def connection_params(self) -> dict:
        """Build Qdrant connection parameters."""
        if self.url:
            params: dict = {"url": self.url, "timeout": self.timeout, "prefer_grpc": self.prefer_grpc}
        else:
            params = {"host": self.host, "port": self.port, "timeout": self.timeout, "prefer_grpc": self.prefer_grpc}

        if self.api_key:
            params["api_key"] = self.api_key.get_secret_value()
        return params


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    model: str = "BAAI/bge-base-en-v1.5"
    dimension: int = 768
    batch_size: int = 64
    device: str = "cpu"
    normalize: bool = True


class RerankerSettings(BaseSettings):
    """FlashRank reranker configuration."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_")

    model: str = "ms-marco-MiniLM-L-12-v2"
    top_n: int = 5
    score_threshold: float = 0.3


class PortkeySettings(BaseSettings):
    """Portkey AI Gateway configuration."""

    model_config = SettingsConfigDict(env_prefix="PORTKEY_")

    api_key: SecretStr = SecretStr("")
    base_url: str = "https://api.portkey.ai/v1"
    virtual_key_primary: Optional[str] = None
    virtual_key_fallback: Optional[str] = None


class LLMSettings(BaseSettings):
    """LLM model configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    primary_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    fallback_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 4096
    request_timeout: int = 60
    max_retries: int = 3


class GuardrailsSettings(BaseSettings):
    """NeMo Guardrails configuration."""

    model_config = SettingsConfigDict(env_prefix="GUARDRAILS_")

    enabled: bool = True
    config_path: str = "src/guardrails/config"
    verbose: bool = False


class GCPSettings(BaseSettings):
    """Google Cloud Platform configuration."""

    model_config = SettingsConfigDict(env_prefix="GCP_")

    project_id: str = ""
    region: str = "us-central1"
    credentials_path: Optional[str] = None


class GCSSettings(BaseSettings):
    """Google Cloud Storage configuration."""

    model_config = SettingsConfigDict(env_prefix="GCS_")

    bucket_name: str = "rag-documents"
    bucket_processed: str = "rag-processed"


class DocumentAISettings(BaseSettings):
    """Google Document AI configuration."""

    model_config = SettingsConfigDict(env_prefix="DOCUMENT_AI_")

    processor_id: Optional[str] = None
    location: str = "us"


class LangSmithSettings(BaseSettings):
    """LangSmith observability configuration."""

    model_config = SettingsConfigDict(env_prefix="LANGCHAIN_")

    tracing_v2: bool = True
    api_key: SecretStr = SecretStr("")
    project: str = "agentic-rag-platform"
    endpoint: str = "https://api.smith.langchain.com"


class LogfireSettings(BaseSettings):
    """Pydantic Logfire configuration."""

    model_config = SettingsConfigDict(env_prefix="LOGFIRE_")

    token: SecretStr = SecretStr("")
    project_name: str = "agentic-rag-platform"
    environment: str = "development"


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""

    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")

    requests: int = 100
    window: int = 60


class MemorySettings(BaseSettings):
    """Conversation memory configuration."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_")

    max_history: int = 10
    summarize_threshold: int = 20


class ChunkingSettings(BaseSettings):
    """Document chunking configuration."""

    model_config = SettingsConfigDict(env_prefix="CHUNK")

    size: int = Field(default=1000, alias="CHUNK_SIZE")
    overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    strategy: str = Field(default="recursive", alias="CHUNKING_STRATEGY")

    model_config = SettingsConfigDict(populate_by_name=True)


class RetrievalSettings(BaseSettings):
    """Retrieval configuration."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")

    top_k: int = 20
    hybrid_dense_weight: float = Field(default=0.7, alias="HYBRID_SEARCH_DENSE_WEIGHT")
    hybrid_sparse_weight: float = Field(default=0.3, alias="HYBRID_SEARCH_SPARSE_WEIGHT")


class CacheSettings(BaseSettings):
    """Semantic cache configuration."""

    model_config = SettingsConfigDict(env_prefix="SEMANTIC_CACHE_")

    enabled: bool = True
    threshold: float = 0.95
    ttl: int = 3600


class Settings(BaseSettings):
    """
    Root application settings.

    All sub-settings are composed here. Access via `get_settings()` singleton.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Environment = Environment.DEVELOPMENT
    app_name: str = "agentic-rag-platform"
    app_version: str = "1.0.0"
    debug: bool = True
    log_level: str = "DEBUG"

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_key: SecretStr = SecretStr("changeme")
    api_key_secondary: Optional[SecretStr] = None
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:8502"]

    # Sub-settings (composed, not from env)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    portkey: PortkeySettings = Field(default_factory=PortkeySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    guardrails: GuardrailsSettings = Field(default_factory=GuardrailsSettings)
    gcp: GCPSettings = Field(default_factory=GCPSettings)
    gcs: GCSSettings = Field(default_factory=GCSSettings)
    document_ai: DocumentAISettings = Field(default_factory=DocumentAISettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    logfire: LogfireSettings = Field(default_factory=LogfireSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"Invalid log_level: {v}. Must be one of {valid}")
        return upper

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Enforce stricter validation in production."""
        if self.app_env == Environment.PRODUCTION:
            if self.debug:
                raise ValueError("DEBUG must be False in production")
            if self.api_key.get_secret_value() == "changeme":
                raise ValueError("Default API key not allowed in production")
            if self.log_level == "DEBUG":
                raise ValueError("DEBUG log level not recommended in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.app_env == Environment.DEVELOPMENT


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings singleton.

    Uses lru_cache to ensure settings are only loaded once.
    Call get_settings.cache_clear() to reload.
    """
    return Settings()
