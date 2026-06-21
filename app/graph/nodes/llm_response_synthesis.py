"""Node: llm_response_synthesis — Pass 2 of the two-pass LLM workflow.

Calls the ResponseSynthesizer with user input, tool name, and tool output
to produce the final natural-language response.

If an error is present in state (set by a previous node), this node generates
a user-facing error message instead of calling the LLM.

Dependencies (injected via make_node):
    synthesizer: ResponseSynthesizer
"""

from typing import Any

from app.graph.state import GraphState
from app.llm.response_synthesizer import ResponseSynthesizer
from app.utils.errors import ResponseSynthesisError
from app.utils.logging import get_logger

log = get_logger(__name__)


def make_node(synthesizer: ResponseSynthesizer):
    """Return the llm_response_synthesis node callable with a bound ResponseSynthesizer.

    Args:
        synthesizer: A fully initialised ResponseSynthesizer instance.
    """

    async def node(state: GraphState) -> dict[str, Any]:
        error = state.get("error")

        # If a previous node raised an error, return a user-facing message.
        if error:
            log.warning("node_response_synthesis_error_passthrough", error=error)
            return {"final_response": f"Sorry, I encountered an error: {error}", "error": None}

        user_input = state.get("user_input", "")
        tool_name = state.get("selected_tool", "unknown")
        tool_output = state.get("tool_output")
        model = state.get("model")

        log.info("node_response_synthesis_start", tool_name=tool_name, model=model)

        try:
            response = await synthesizer.synthesize(
                user_input=user_input,
                tool_name=tool_name,
                tool_output=tool_output,
                model=model,
            )
        except ResponseSynthesisError as exc:
            log.error("node_response_synthesis_failed", error=str(exc))
            return {"error": str(exc)}

        log.info("node_response_synthesis_complete", length=len(response))
        return {"final_response": response, "error": None}

    return node
