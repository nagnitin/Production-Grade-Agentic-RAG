"""
LangGraph StateGraph — The agentic RAG pipeline.

WHY: LangGraph provides a state machine abstraction that gives us:
1. Deterministic control flow with conditional routing
2. Typed state that flows through all nodes
3. Built-in checkpointing for human-in-the-loop
4. Retry and error recovery at the graph level
5. Visual graph representation for debugging

ARCHITECTURE DECISION: This is a DAG with conditional edges, not a simple
linear chain. The planner routes queries to different sub-paths:

    START → Planner ──→ Retriever → Reranker → Responder → Memory → END
                   └──→ Responder → Memory → END  (chitchat path)

Each node is an async function that reads from and writes to the typed state.
The graph is compiled once and reused across all requests.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.edges import (
    route_after_planner,
    route_after_reranker,
    route_after_responder,
    route_after_retriever,
    should_end,
)
from src.agent.nodes.memory import memory_node
from src.agent.nodes.planner import planner_node
from src.agent.nodes.reranker import reranker_node
from src.agent.nodes.responder import responder_node
from src.agent.nodes.retriever import retriever_node
from src.agent.state import AgentState
from src.config.constants import NodeName
from src.config.logging_config import get_logger

logger = get_logger(__name__)


def build_agent_graph(
    checkpointer: Any = None,
) -> StateGraph:
    """
    Build and compile the LangGraph agent.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence.
            Use PostgresSaver for production, MemorySaver for testing.

    Returns:
        Compiled StateGraph ready for invocation.

    Graph Structure:
        START
          │
          ▼
        Planner
          │
          ├──[intent=rag]──→ Retriever
          │                      │
          │                      ▼
          │                  Reranker
          │                      │
          │                      ▼
          └──[else]────────→ Responder
                                 │
                                 ▼
                              Memory
                                 │
                                 ▼
                                END
    """
    logger.info("Building agent graph")

    # === Create the StateGraph ===
    graph = StateGraph(AgentState)

    # === Add Nodes ===
    graph.add_node(NodeName.PLANNER, planner_node)
    graph.add_node(NodeName.RETRIEVER, retriever_node)
    graph.add_node(NodeName.RERANKER, reranker_node)
    graph.add_node(NodeName.RESPONDER, responder_node)
    graph.add_node(NodeName.MEMORY, memory_node)

    # === Set Entry Point ===
    graph.set_entry_point(NodeName.PLANNER)

    # === Add Conditional Edges ===

    # After Planner: route based on intent
    graph.add_conditional_edges(
        NodeName.PLANNER,
        route_after_planner,
        {
            NodeName.RETRIEVER: NodeName.RETRIEVER,
            NodeName.RESPONDER: NodeName.RESPONDER,
        },
    )

    # After Retriever: route based on results
    graph.add_conditional_edges(
        NodeName.RETRIEVER,
        route_after_retriever,
        {
            NodeName.RERANKER: NodeName.RERANKER,
            NodeName.RETRIEVER: NodeName.RETRIEVER,  # Retry
            NodeName.RESPONDER: NodeName.RESPONDER,  # Empty results
        },
    )

    # After Reranker: always to responder
    graph.add_conditional_edges(
        NodeName.RERANKER,
        route_after_reranker,
        {
            NodeName.RESPONDER: NodeName.RESPONDER,
        },
    )

    # After Responder: always to memory
    graph.add_conditional_edges(
        NodeName.RESPONDER,
        route_after_responder,
        {
            NodeName.MEMORY: NodeName.MEMORY,
        },
    )

    # After Memory: end
    graph.add_conditional_edges(
        NodeName.MEMORY,
        should_end,
        {
            "__end__": END,
        },
    )

    # === Compile ===
    compile_kwargs: dict[str, Any] = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    compiled = graph.compile(**compile_kwargs)

    logger.info("Agent graph compiled successfully")
    return compiled


async def run_agent(
    graph: Any,
    query: str,
    session_id: str = "default",
    user_id: str = "anonymous",
    filters: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute the agent graph with a user query.

    This is the main entry point for running the agent. It prepares
    the initial state and invokes the compiled graph.

    Args:
        graph: Compiled LangGraph StateGraph
        query: User's question
        session_id: Conversation session ID
        user_id: User identifier
        filters: Optional metadata filters for retrieval
        config: Runtime configuration (gateway, vectorstore, etc.)

    Returns:
        Final agent state with response, citations, confidence, and metadata.
    """
    initial_state: AgentState = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "filters": filters or {},
        "messages": [],
        "chat_history": [],
        "intent": "",
        "retrieval_strategy": "",
        "query_decomposition": [],
        "documents": [],
        "reranked_documents": [],
        "response": "",
        "citations": [],
        "confidence": 0.0,
        "guardrails_input_result": None,
        "guardrails_output_result": None,
        "guardrails_blocked": False,
        "metadata": {
            "node_execution_order": [],
            "errors": [],
        },
        "error": None,
        "retry_count": 0,
        "should_retry": False,
    }

    run_config = {
        "configurable": {
            **(config or {}),
            "thread_id": session_id,
        },
    }

    logger.info(
        "Running agent",
        query=query[:100],
        session_id=session_id,
        user_id=user_id,
    )

    result = await graph.ainvoke(initial_state, config=run_config)

    logger.info(
        "Agent execution complete",
        intent=result.get("intent"),
        confidence=result.get("confidence"),
        citations_count=len(result.get("citations", [])),
        execution_order=result.get("metadata", {}).get("node_execution_order", []),
    )

    return result
