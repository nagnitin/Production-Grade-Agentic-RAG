"""
LangSmith Tracing configuration.

WHY: Enterprise RAG platforms require trace visibility for complex multi-node agent runs.
This module configures the global LangChain environment variables to enable LangSmith.
"""

from __future__ import annotations

import os

from src.config.logging_config import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


def setup_langsmith(settings: Settings) -> None:
    """
    Set up LangSmith tracing environment variables.
    
    LangChain automatically reads environment variables prefixed with LANGCHAIN_
    to configure and stream trace metrics to the cloud.
    """
    api_key_secret = settings.langsmith.api_key
    
    if not api_key_secret or not api_key_secret.get_secret_value():
        logger.info("LangSmith API key not configured. Tracing is disabled.")
        return

    # Export configurations to standard LangChain environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langsmith.tracing_v2).lower()
    os.environ["LANGCHAIN_API_KEY"] = api_key_secret.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith.project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith.endpoint

    logger.info(
        "LangSmith tracing enabled",
        project=settings.langsmith.project,
        endpoint=settings.langsmith.endpoint,
    )
