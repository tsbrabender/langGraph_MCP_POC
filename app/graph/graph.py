"""LangGraph graph assembly and compilation.

Provides two graph variants:
  - build_llm_graph():    Fully LLM-driven 5-node linear flow.
  - build_hybrid_graph(): Adds keyword intent classification before LLM selection.

Both graphs accept the same dependency set (selector, executor, synthesizer)
and return compiled LangGraph runnables that can be invoked with ainvoke().

The optional db_client parameter (a SQLiteClient) enables workflow history
persistence inside the finalize_response node.

Usage:
    from app.graph.graph import build_llm_graph
    from app.llm.tool_selector import ToolSelector
    from app.llm.response_synthesizer import ResponseSynthesizer
    from app.services.mcp_executor import MCPExecutor

    graph = build_llm_graph(selector, executor, synthesizer)
    result = await graph.ainvoke({"user_input": "list the files"})
"""

from typing import Any

from langgraph.graph import END, StateGraph

from app.graph.edges.routing import (
    after_intent_classification,
    after_tool_invocation,
    after_tool_selection,
)
from app.graph.nodes import (
    classify_intent,
    finalize_response,
    ingest_user_input,
    llm_response_synthesis,
    llm_tool_selection,
    mcp_tool_invocation,
)
from app.graph.state import GraphState
from app.utils.logging import get_logger

log = get_logger(__name__)


def build_llm_graph(
    selector: Any,
    executor: Any,
    synthesizer: Any,
    db_client: Any | None = None,
):
    """Build and compile the fully LLM-driven workflow graph.

    Flow:
        ingest_user_input
            → llm_tool_selection
                → [error] → finalize_response → END
                → [ok]    → mcp_tool_invocation
                                → llm_response_synthesis
                                    → finalize_response → END

    Args:
        selector:    A ToolSelector instance.
        executor:    An MCPExecutor instance.
        synthesizer: A ResponseSynthesizer instance.
        db_client:   Optional SQLiteClient for workflow history persistence.

    Returns:
        A compiled LangGraph runnable.
    """
    workflow: StateGraph = StateGraph(GraphState)

    # Nodes
    workflow.add_node("ingest_user_input", ingest_user_input.make_node())
    workflow.add_node("llm_tool_selection", llm_tool_selection.make_node(selector))
    workflow.add_node("mcp_tool_invocation", mcp_tool_invocation.make_node(executor))
    workflow.add_node("llm_response_synthesis", llm_response_synthesis.make_node(synthesizer))
    workflow.add_node("finalize_response", finalize_response.make_node(db_client=db_client))

    # Edges
    workflow.set_entry_point("ingest_user_input")
    workflow.add_edge("ingest_user_input", "llm_tool_selection")
    workflow.add_conditional_edges(
        "llm_tool_selection",
        after_tool_selection,
        {
            "mcp_tool_invocation": "mcp_tool_invocation",
            "finalize_response": "finalize_response",
        },
    )
    workflow.add_conditional_edges(
        "mcp_tool_invocation",
        after_tool_invocation,
        {"llm_response_synthesis": "llm_response_synthesis"},
    )
    workflow.add_edge("llm_response_synthesis", "finalize_response")
    workflow.add_edge("finalize_response", END)

    log.info("graph_compiled", mode="llm")
    return workflow.compile()


def build_hybrid_graph(
    selector: Any,
    executor: Any,
    synthesizer: Any,
    db_client: Any | None = None,
):
    """Build and compile the hybrid workflow graph with keyword intent classification.

    Flow:
        ingest_user_input
            → classify_intent
                → [full match: selected_tool set] → mcp_tool_invocation
                                                        → llm_response_synthesis
                                                            → finalize_response → END
                → [no full match]                 → llm_tool_selection
                                                        → [error] → finalize_response → END
                                                        → [ok]    → mcp_tool_invocation
                                                                        → llm_response_synthesis
                                                                            → finalize_response → END

    When classify_intent sets only intent (tool name but no arguments), llm_tool_selection
    receives the intent as a context hint to constrain the LLM's choice.

    Args:
        selector:    A ToolSelector instance.
        executor:    An MCPExecutor instance.
        synthesizer: A ResponseSynthesizer instance.
        db_client:   Optional SQLiteClient for workflow history persistence.

    Returns:
        A compiled LangGraph runnable.
    """
    workflow: StateGraph = StateGraph(GraphState)

    # Nodes
    workflow.add_node("ingest_user_input", ingest_user_input.make_node())
    workflow.add_node("classify_intent", classify_intent.make_node())
    workflow.add_node("llm_tool_selection", llm_tool_selection.make_node(selector))
    workflow.add_node("mcp_tool_invocation", mcp_tool_invocation.make_node(executor))
    workflow.add_node("llm_response_synthesis", llm_response_synthesis.make_node(synthesizer))
    workflow.add_node("finalize_response", finalize_response.make_node(db_client=db_client))

    # Edges
    workflow.set_entry_point("ingest_user_input")
    workflow.add_edge("ingest_user_input", "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        after_intent_classification,
        {
            "mcp_tool_invocation": "mcp_tool_invocation",
            "llm_tool_selection": "llm_tool_selection",
        },
    )
    workflow.add_conditional_edges(
        "llm_tool_selection",
        after_tool_selection,
        {
            "mcp_tool_invocation": "mcp_tool_invocation",
            "finalize_response": "finalize_response",
        },
    )
    workflow.add_conditional_edges(
        "mcp_tool_invocation",
        after_tool_invocation,
        {"llm_response_synthesis": "llm_response_synthesis"},
    )
    workflow.add_edge("llm_response_synthesis", "finalize_response")
    workflow.add_edge("finalize_response", END)

    log.info("graph_compiled", mode="hybrid")
    return workflow.compile()
