"""Shared graph state — TypedDict schema passed between all LangGraph nodes.

LangGraph works natively with TypedDict. All fields are optional (total=False)
so nodes may return partial updates; absent fields are treated as unset.

The only field that must be present in the initial invocation dict is user_input.
"""

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    # --- Core workflow fields ---
    user_input: str            # The raw (then normalized) user request
    selected_tool: str         # Tool name chosen by LLM or intent classifier
    tool_arguments: dict[str, Any]  # Validated arguments for the selected tool
    tool_output: Any           # Structured output returned by the MCP tool
    final_response: str        # Natural-language response for the user

    # --- Routing ---
    intent: str                # Keyword-classified tool name (hybrid mode only)

    # --- Conversation memory (Step 6) ---
    conversation_history: list[dict[str, Any]]

    # --- Error propagation ---
    error: str                 # Set by any node on failure; triggers error routing

    # --- Metadata ---
    metadata: dict[str, Any]   # request_id, timestamps, etc.
