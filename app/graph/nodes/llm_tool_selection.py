"""Node: llm_tool_selection — Pass 1 of the two-pass LLM workflow.

Calls the ToolSelector with the current user input and any pre-classified intent,
then writes the validated tool name and arguments back to state.

Dependencies (injected via make_node):
    selector: ToolSelector
"""

from typing import Any

from app.graph.state import GraphState
from app.llm.tool_selector import ToolSelector
from app.utils.errors import ToolSelectionError
from app.utils.logging import get_logger

log = get_logger(__name__)


def make_node(selector: ToolSelector):
    """Return the llm_tool_selection node callable with a bound ToolSelector.

    Args:
        selector: A fully initialised ToolSelector instance.
    """

    async def node(state: GraphState) -> dict[str, Any]:
        user_input = state.get("user_input", "")
        intent = state.get("intent")
        model = state.get("model")
        log.info("node_tool_selection_start", user_input=user_input[:80], intent=intent, model=model)

        context: dict[str, Any] | None = None
        if intent:
            context = {"intent": intent, "hint": f"Prefer the '{intent}' tool."}

        try:
            tool_call = await selector.select(user_input, context=context, model=model)
        except ToolSelectionError as exc:
            log.error("node_tool_selection_failed", error=str(exc))
            return {"error": str(exc)}

        log.info(
            "node_tool_selection_complete",
            tool_name=tool_call.tool_name,
            reasoning=tool_call.reasoning,
        )
        return {
            "selected_tool": tool_call.tool_name,
            "tool_arguments": tool_call.arguments,
            "error": None,
        }

    return node
