"""Conditional edge routing functions for the LangGraph workflows.

Each function takes the current GraphState and returns a string that is
the name of the next node to execute. LangGraph routes accordingly.
"""

from app.graph.state import GraphState


def after_tool_selection(state: GraphState) -> str:
    """Route after llm_tool_selection.

    - On error → finalize_response (skip invocation and synthesis)
    - On success → mcp_tool_invocation
    """
    return "finalize_response" if state.get("error") else "mcp_tool_invocation"


def after_tool_invocation(state: GraphState) -> str:
    """Route after mcp_tool_invocation.

    - On error → llm_response_synthesis (which converts the error to a user message)
    - On success → llm_response_synthesis
    """
    # Always proceed to synthesis; llm_response_synthesis handles error state.
    return "llm_response_synthesis"


def after_intent_classification(state: GraphState) -> str:
    """Route after classify_intent (hybrid graph only).

    - If selected_tool is already set by the classifier (full extraction) →
        mcp_tool_invocation (skip LLM selection entirely)
    - Otherwise → llm_tool_selection (full LLM selection, optionally with intent hint)
    """
    return "mcp_tool_invocation" if state.get("selected_tool") else "llm_tool_selection"
