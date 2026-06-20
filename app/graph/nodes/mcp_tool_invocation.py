"""Node: mcp_tool_invocation — executes the selected MCP tool via MCPExecutor.

Reads selected_tool and tool_arguments from state, calls the executor,
and writes tool_output back to state.

Dependencies (injected via make_node):
    executor: MCPExecutor
"""

from typing import Any

from app.graph.state import GraphState
from app.services.mcp_executor import MCPExecutor
from app.utils.logging import get_logger

log = get_logger(__name__)


def make_node(executor: MCPExecutor):
    """Return the mcp_tool_invocation node callable with a bound MCPExecutor.

    Args:
        executor: A fully initialised MCPExecutor instance.
    """

    async def node(state: GraphState) -> dict[str, Any]:
        tool_name = state.get("selected_tool")
        tool_args = state.get("tool_arguments") or {}

        if not tool_name:
            msg = "mcp_tool_invocation: no selected_tool in state"
            log.error(msg)
            return {"error": msg}

        log.info("node_tool_invocation_start", tool_name=tool_name, args=tool_args)

        try:
            output = await executor.execute(tool_name, tool_args)
        except Exception as exc:
            log.error("node_tool_invocation_failed", tool_name=tool_name, error=str(exc))
            return {"error": str(exc)}

        log.info("node_tool_invocation_complete", tool_name=tool_name)
        return {"tool_output": output, "error": None}

    return node
