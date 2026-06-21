"""MCPExecutor — dispatches ToolCall objects to MCP tool implementations.

This service is the only non-node layer that imports from app/mcp_server/tools/.
Graph nodes receive an MCPExecutor instance via dependency injection and never
import MCP tool modules directly.
"""

from pathlib import Path
from typing import Any

from app.mcp_server.tools import (
    extract_metadata,
    fetch_web_resource,
    get_cached_resource,
    get_topic_resources,
    list_files,
    read_file,
    refresh_cache,
    search_files,
    summarize_file,
)
from app.utils.errors import MCPToolError
from app.utils.logging import get_logger

log = get_logger(__name__)


class MCPExecutor:
    """Dispatches (tool_name, arguments) to the appropriate MCP tool implementation.

    Args:
        sandbox_root: Absolute path to the file-system sandbox.
        llm_client:   OllamaClient instance, required only for the summarize_file tool.
        cache_client: Cache backend (SQLiteCacheClient), required for context retrieval tools.
    """

    def __init__(
        self,
        sandbox_root: Path,
        llm_client: Any | None = None,
        cache_client: Any | None = None,
    ) -> None:
        self._sandbox_root = sandbox_root
        self._llm = llm_client
        self._cache = cache_client

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call and return its output as a plain dict.

        Args:
            tool_name: Exact name of the registered MCP tool.
            arguments: Validated argument dict for the tool.

        Returns:
            The tool's Pydantic output model serialized via model_dump().

        Raises:
            MCPToolError: If the tool name is unknown or execution fails.
        """
        log.info("mcp_executor_dispatch", tool_name=tool_name, arguments=arguments)
        match tool_name:
            case "list_files":
                result = await list_files.run(
                    directory=arguments.get("directory", "."),
                    sandbox_root=self._sandbox_root,
                )
            case "read_file":
                result = await read_file.run(
                    path=arguments["path"],
                    sandbox_root=self._sandbox_root,
                )
            case "search_files":
                result = await search_files.run(
                    pattern=arguments["pattern"],
                    directory=arguments.get("directory", "."),
                    sandbox_root=self._sandbox_root,
                )
            case "extract_metadata":
                result = await extract_metadata.run(
                    path=arguments["path"],
                    sandbox_root=self._sandbox_root,
                )
            case "summarize_file":
                if self._llm is None:
                    raise MCPToolError(
                        "summarize_file requires an LLM client but none was provided to MCPExecutor"
                    )
                result = await summarize_file.run(
                    path=arguments["path"],
                    sandbox_root=self._sandbox_root,
                    llm=self._llm,
                )
            case "get_topic_resources":
                result = await get_topic_resources.run(topic=arguments["topic"])
            case "fetch_web_resource":
                result = await fetch_web_resource.run(url=arguments["url"])
            case "get_cached_resource":
                if self._cache is None:
                    raise MCPToolError(
                        "get_cached_resource requires a cache client but none was provided to MCPExecutor"
                    )
                result = await get_cached_resource.run(
                    url=arguments["url"],
                    cache_client=self._cache,
                )
            case "refresh_cache":
                if self._cache is None:
                    raise MCPToolError(
                        "refresh_cache requires a cache client but none was provided to MCPExecutor"
                    )
                result = await refresh_cache.run(
                    url=arguments["url"],
                    ttl_seconds=arguments.get("ttl_seconds", 21600),
                    cache_client=self._cache,
                    fetcher=fetch_web_resource.run,
                )
            case _:
                raise MCPToolError(f"Unknown tool: '{tool_name}'")

        output = result.model_dump()
        log.info("mcp_executor_complete", tool_name=tool_name)
        return output
