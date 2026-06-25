"""
Planner Node — Intent classification, query understanding, and retrieval strategy.

WHY: Not every query needs full RAG retrieval. A planner node reduces latency
and cost by routing simple queries (chitchat, clarification) directly to the
responder, bypassing retrieval and reranking entirely.

ARCHITECTURE DECISION: Using LLM-based classification (few-shot) instead of a
fine-tuned classifier because:
1. Zero training data needed — works immediately
2. Adapts to new intents via prompt changes
3. Handles nuance better than keyword matching
4. Classification quality improves with the same model upgrades

TRADE-OFF: LLM classification adds ~200-500ms latency vs. a fine-tuned classifier
(~10ms). Acceptable for enterprise use where accuracy > speed for routing.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.agent.prompts.system_prompts import PLANNER_SYSTEM_PROMPT
from src.agent.state import AgentState
from src.config.constants import Intent, RetrievalStrategy
from src.config.logging_config import get_logger

logger = get_logger(__name__)


async def planner_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Classify user intent and determine retrieval strategy.

    Reads: query, chat_history
    Writes: intent, retrieval_strategy, query_decomposition, metadata

    The planner uses few-shot LLM classification to categorize the query
    and select the optimal retrieval strategy.
    """
    start_time = time.perf_counter()
    query = state["query"]
    chat_history = state.get("chat_history", [])

    logger.info("Planner executing", query=query[:100])

    try:
        # Get the LLM gateway from config
        gateway = config["configurable"]["gateway"]

        # Check if using the rate-limited apifreellm gateway.
        # If so, bypass the planner LLM call to save rate-limits for the responder.
        import os
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        portkey_api_key = gateway.settings.portkey.api_key.get_secret_value()
        is_openai_unconfigured = not openai_api_key or openai_api_key == "your-openai-api-key-here"
        is_portkey_unconfigured = not portkey_api_key or portkey_api_key == "changeme"

        if is_openai_unconfigured and is_portkey_unconfigured:
            logger.info("Bypassing planner LLM call to conserve apifreellm API rate limit.")
            return {
                "intent": Intent.RAG,
                "retrieval_strategy": RetrievalStrategy.HYBRID,
                "query_decomposition": [query],
                "metadata": {
                    **state.get("metadata", {}),
                    "node_execution_order": state.get("metadata", {}).get(
                        "node_execution_order", []
                    ) + ["planner"],
                },
            }

        # Build the classification prompt with conversation context
        context_str = ""
        if chat_history:
            recent = chat_history[-4:]  # Last 2 turns
            context_str = "\n".join(
                f"{'User' if i % 2 == 0 else 'Assistant'}: {msg.content}"
                for i, msg in enumerate(recent)
            )
            context_str = f"\nRecent conversation:\n{context_str}\n"

        classification_prompt = (
            f"{context_str}"
            f"\nUser Query: {query}\n\n"
            f"Classify this query and determine the retrieval strategy. "
            f"Respond with valid JSON only."
        )

        # Call LLM for classification
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=classification_prompt),
        ]

        response = await gateway.ainvoke(
            messages,
            temperature=0.0,  # Deterministic classification
            max_tokens=256,
        )

        # Parse the classification result
        result = _parse_classification(response.content, query)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Planner completed",
            intent=result["intent"],
            strategy=result["retrieval_strategy"],
            latency_ms=round(latency_ms, 2),
        )

        return {
            "intent": result["intent"],
            "retrieval_strategy": result["retrieval_strategy"],
            "query_decomposition": result.get("query_decomposition", [query]),
            "metadata": {
                **state.get("metadata", {}),
                "node_execution_order": state.get("metadata", {}).get(
                    "node_execution_order", []
                ) + ["planner"],
            },
        }

    except Exception as e:
        logger.error("Planner failed", error=str(e), query=query[:100])
        # Default to RAG with hybrid search on failure — safe fallback
        return {
            "intent": Intent.RAG,
            "retrieval_strategy": RetrievalStrategy.HYBRID,
            "query_decomposition": [query],
            "error": f"Planner error: {str(e)}",
            "metadata": {
                **state.get("metadata", {}),
                "errors": state.get("metadata", {}).get("errors", [])
                + [f"planner: {str(e)}"],
                "node_execution_order": state.get("metadata", {}).get(
                    "node_execution_order", []
                ) + ["planner"],
            },
        }


def _parse_classification(response_text: str, original_query: str) -> dict[str, Any]:
    """
    Parse LLM classification response into structured output.

    Handles:
    - Clean JSON responses
    - JSON wrapped in markdown code blocks
    - Malformed responses (falls back to RAG)
    """
    text = response_text.strip()

    # Remove markdown code block wrappers if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse planner JSON, defaulting to RAG", raw=text[:200])
        return {
            "intent": Intent.RAG,
            "retrieval_strategy": RetrievalStrategy.HYBRID,
            "query_decomposition": [original_query],
        }

    # Validate intent
    valid_intents = {Intent.RAG, Intent.CHITCHAT, Intent.CLARIFICATION, Intent.OUT_OF_SCOPE}
    intent = parsed.get("intent", Intent.RAG)
    if intent not in valid_intents:
        intent = Intent.RAG

    # Validate retrieval strategy
    valid_strategies = {RetrievalStrategy.DENSE, RetrievalStrategy.SPARSE, RetrievalStrategy.HYBRID}
    strategy = parsed.get("retrieval_strategy", RetrievalStrategy.HYBRID)
    if strategy not in valid_strategies:
        strategy = RetrievalStrategy.HYBRID

    # Parse query decomposition
    decomposition = parsed.get("query_decomposition", [original_query])
    if not isinstance(decomposition, list) or not decomposition:
        decomposition = [original_query]

    return {
        "intent": intent,
        "retrieval_strategy": strategy,
        "query_decomposition": decomposition,
        "reasoning": parsed.get("reasoning", ""),
    }
