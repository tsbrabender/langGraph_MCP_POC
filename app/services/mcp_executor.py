"""MCPExecutor — dispatches ToolCall objects via the live ToolRegistry.

All tool lookup and dependency injection is driven by the ToolRegistry.
No hardcoded match statement; adding or removing a tool only requires a
registry.reload() call — no changes to this file.
"""

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.utils.errors import MCPToolError
from app.utils.logging import get_logger

if TYPE_CHECKING:
    from app.llm.tool_registry import ToolRegistry

log = get_logger(__name__)

# Handler parameter names the executor knows how to inject automatically.
_SANDBOX_PARAM = "sandbox_root"
_LLM_PARAM = "llm"
_CACHE_PARAM = "cache_client"


class MCPExecutor:
    """Dispatches (tool_name, arguments) to the appropriate MCP tool handler.

    Reads from the live ToolRegistry on every execute() call, so tools added
    or removed via registry.reload() are reflected immediately.

    Dependency injection rules (applied to the handler's signature):
      - "sandbox_root" parameter → receives self._sandbox_root (Path)
      - "llm" parameter          → receives self._llm (OllamaClient)
      - "cache_client" parameter → receives self._cache (SQLiteCacheClient)
      - ToolSpec.dependencies    → resolved to the named tool's handler function

    Args:
        sandbox_root:  Absolute path to the file-system sandbox.
        llm_client:    OllamaClient instance; required for LLM-powered tools.
        cache_client:  Cache backend; required for context-retrieval tools.
        tool_registry: Live ToolRegistry; consulted on every execute() call.
    """

    def __init__(
        self,
        sandbox_root: Path,
        llm_client: Any | None = None,
        cache_client: Any | None = None,
        tool_registry: "ToolRegistry | None" = None,
    ) -> None:
        self._sandbox_root = sandbox_root
        self._llm = llm_client
        self._cache = cache_client
        self._registry = tool_registry

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call and return its output as a plain dict.

        Args:
            tool_name:  Exact name of the registered MCP tool.
            arguments:  Pre-validated argument dict.

        Returns:
            The tool's Pydantic output model serialized via model_dump().

        Raises:
            MCPToolError: Unknown tool, missing required dependency, or execution error.
        """
        if self._registry is None:
            raise MCPToolError("MCPExecutor has no ToolRegistry configured.")

        specs = self._registry.specs_by_name
        spec = specs.get(tool_name)
        if spec is None:
            raise MCPToolError(
                f"Unknown tool: '{tool_name}'. "
                f"Known tools: {sorted(specs.keys())}"
            )

        log.info(
            "mcp_executor_dispatch",
            tool_name=tool_name,
            category=spec.category,
            arguments=arguments,
        )

        # Start with the caller-supplied arguments.
        kwargs: dict[str, Any] = dict(arguments)

        # Auto-inject infrastructure dependencies based on the handler's signature.
        sig_params = inspect.signature(spec.handler).parameters

        if _SANDBOX_PARAM in sig_params:
            kwargs[_SANDBOX_PARAM] = self._sandbox_root

        if _LLM_PARAM in sig_params:
            if self._llm is None:
                raise MCPToolError(
                    f"Tool '{tool_name}' requires an LLM client but none was configured in MCPExecutor."
                )
            kwargs[_LLM_PARAM] = self._llm

        if _CACHE_PARAM in sig_params:
            if self._cache is None:
                raise MCPToolError(
                    f"Tool '{tool_name}' requires a cache client but none was configured in MCPExecutor."
                )
            kwargs[_CACHE_PARAM] = self._cache

        # Resolve inter-tool dependencies declared in ToolSpec.dependencies.
        # Example: refresh_cache declares {"fetcher": "fetch_web_resource"}, which
        # resolves to the fetch_web_resource handler function.
        for param_name, dep_tool_name in spec.dependencies.items():
            dep_spec = specs.get(dep_tool_name)
            if dep_spec is None:
                raise MCPToolError(
                    f"Tool '{tool_name}' declares a dependency on '{dep_tool_name}' "
                    f"which is not currently registered. Known tools: {sorted(specs.keys())}"
                )
            kwargs[param_name] = dep_spec.handler

        try:
            result = await spec.handler(**kwargs)
        except MCPToolError:
            raise
        except Exception as exc:
            log.error(
                "mcp_executor_tool_error",
                tool_name=tool_name,
                category=spec.category,
                error=str(exc),
                exc_info=True,
            )
            raise MCPToolError(f"Tool '{tool_name}' raised an unexpected error: {exc}") from exc

        output = result.model_dump()
        log.info("mcp_executor_complete", tool_name=tool_name)
        return output
