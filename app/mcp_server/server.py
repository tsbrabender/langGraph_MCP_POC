"""FastMCP server entry point — registers all MCP tools via dynamic discovery.

Run with:
    python -m app.mcp_server.server
"""

import inspect
from pathlib import Path

from fastmcp import FastMCP

from app.llm.ollama_client import OllamaClient
from app.mcp_server.tool_loader import discover_all_tools
from app.mcp_server.tool_spec import ToolSpec
from app.utils.config import get_settings
from app.utils.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger(__name__)

mcp = FastMCP(
    name="langgraph-mcp-poc",
    instructions=(
        "File-system agent tools with sandboxed access. "
        f"All paths are relative to the sandbox root: {settings.sandbox_root}"
    ),
)

_sandbox_root = Path(settings.sandbox_root).resolve()
_sandbox_root.mkdir(parents=True, exist_ok=True)

_llm = OllamaClient()


def _make_mcp_wrapper(spec: ToolSpec, sandbox_root: Path, llm: OllamaClient):
    """Return an async wrapper function suitable for registration with FastMCP.

    Injects sandbox_root and llm dependencies based on the handler's signature
    so callers only need to pass the tool's declared input arguments.
    """
    sig_params = inspect.signature(spec.handler).parameters
    needs_sandbox = "sandbox_root" in sig_params
    needs_llm = "llm" in sig_params

    async def wrapper(**kwargs):
        if needs_sandbox:
            kwargs["sandbox_root"] = sandbox_root
        if needs_llm:
            kwargs["llm"] = llm
        result = await spec.handler(**kwargs)
        return result.model_dump()

    wrapper.__name__ = spec.name
    wrapper.__doc__ = spec.description
    return wrapper


# ---------------------------------------------------------------------------
# Dynamic tool registration — discovers all tools at server startup.
# ---------------------------------------------------------------------------

_all_tools = discover_all_tools()
_total = sum(len(v) for v in _all_tools.values())

for _category_name, _specs in _all_tools.items():
    for _spec in _specs:
        _wrapped = _make_mcp_wrapper(_spec, _sandbox_root, _llm)
        mcp.tool()(_wrapped)
        log.info("mcp_tool_registered", category=_category_name, tool=_spec.name)

log.info(
    "mcp_tools_registration_complete",
    categories=list(_all_tools.keys()),
    total=_total,
)


if __name__ == "__main__":
    log.info(
        "starting_mcp_server",
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
        sandbox_root=str(_sandbox_root),
        total_tools=_total,
    )
    mcp.run(
        transport="streamable-http",
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
    )
