"""
Golden datasets for RAGAS evaluation.

WHY: Enterprise RAG evaluation requires standard, high-quality test queries and ground truths.
This module defines the golden datasets used to calculate faithfulness, answer relevance,
and context precision metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Golden datasets containing reference QA pairs
GOLDEN_DATASETS: Dict[str, List[Dict[str, Any]]] = {
    "default": [
        {
            "question": "What document formats are supported by the ingestion pipeline?",
            "ground_truth": (
                "The ingestion pipeline supports PDF (.pdf), DOCX (.docx), PPTX (.pptx), "
                "HTML (.html, .htm), and TXT (.txt) formats. The maximum allowed file size is 16MB."
            ),
        },
        {
            "question": "How does semantic caching optimize search latency?",
            "ground_truth": (
                "Semantic caching stores previously generated LLM answers mapped to query embeddings. "
                "When a new query has high cosine similarity (above 0.5) to a cached query, "
                "the cached answer is returned immediately, bypassing LLM execution."
            ),
        },
        {
            "question": "What happens if a query fails input guardrails?",
            "ground_truth": (
                "If a user query fails input guardrails (e.g. contains PII like credit cards or jailbreak attempts), "
                "it is blocked immediately. The system logs a warning, increments the error count, "
                "and returns a standard safety refusal message."
            ),
        },
        {
            "question": "What is the embedding model used for vector storage?",
            "ground_truth": (
                "The system uses SentenceTransformers or fastembed embedding models. By default, "
                "it uses 'BAAI/bge-small-en-v1.5' for generating document and query embeddings."
            ),
        },
        {
            "question": "How is history managed in the agentic memory layer?",
            "ground_truth": (
                "Session history is persisted in a PostgreSQL database using LangGraph's "
                "Postgres checkpoint connection. This enables stateful, multi-turn conversations "
                "linked to unique session IDs."
            ),
        }
    ],
    "security": [
        {
            "question": "Does the system log raw API keys?",
            "ground_truth": "No, all sensitive fields including Portkey API keys, database credentials, and GCP keys are loaded as Pydantic SecretStr and masked in structured logs.",
        },
        {
            "question": "What PII types are censored in responses?",
            "ground_truth": "The output guardrails mask email addresses, phone numbers, social security numbers (SSN), and credit card numbers.",
        }
    ]
}


def get_dataset(name: str = "default") -> List[Dict[str, Any]]:
    """Retrieve a dataset by name, falling back to 'default'."""
    return GOLDEN_DATASETS.get(name, GOLDEN_DATASETS["default"])
