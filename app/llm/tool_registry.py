"""Tool registry — dynamic, thread-safe bridge between the LLM layer and MCP tools.

ToolRegistry replaces the static build_tool_definitions() function. It wraps
discover_all_tools() and exposes the current ToolDefinitions and ToolSpecs to
ToolSelector and MCPExecutor. Both classes hold a reference to the same
ToolRegistry instance, so a single registry.reload() propagates immediately
to all consumers without rebuilding the graph.
"""

import threading

from app.llm.tool_selector import ToolDefinition
from app.mcp_server.tool_loader import discover_all_tools
from app.mcp_server.tool_spec import ToolSpec
from app.utils.logging import get_logger

log = get_logger(__name__)


class ToolRegistry:
    """Thread-safe, hot-reloadable registry of MCP tool specs and LLM definitions.

    Usage:
        registry = ToolRegistry()
        registry.reload()               # populate at startup
        ...
        categories = registry.reload()  # refresh at runtime (POST /api/tools/refresh)

    ToolSelector and MCPExecutor each hold a reference to this registry and
    always read the current state on every operation — no server restart needed.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._definitions: list[ToolDefinition] = []
        self._specs_by_name: dict[str, ToolSpec] = {}
        self._categories: dict[str, list[str]] = {}

    def reload(self) -> dict[str, list[str]]:
        """Discover all tools from disk and replace the in-memory registry.

        Returns:
            dict mapping category_name → [tool_name, ...] for every loaded tool.
        """
        raw: dict[str, list[ToolSpec]] = discover_all_tools()

        definitions: list[ToolDefinition] = []
        specs_by_name: dict[str, ToolSpec] = {}
        categories: dict[str, list[str]] = {}

        for cat_name, specs in raw.items():
            categories[cat_name] = []
            for spec in specs:
                definitions.append(
                    ToolDefinition(
                        name=spec.name,
                        description=spec.description,
                        input_schema_class=spec.input_schema_class,
                    )
                )
                specs_by_name[spec.name] = spec
                categories[cat_name].append(spec.name)
                log.info("tool_registered", category=cat_name, tool=spec.name)

        with self._lock:
            self._definitions = definitions
            self._specs_by_name = specs_by_name
            self._categories = categories

        log.info(
            "registry_reloaded",
            total_tools=len(definitions),
            categories=list(categories.keys()),
        )
        return categories

    @property
    def definitions(self) -> list[ToolDefinition]:
        """Return the current list of ToolDefinitions (snapshot under lock)."""
        with self._lock:
            return list(self._definitions)

    @property
    def specs_by_name(self) -> dict[str, ToolSpec]:
        """Return the current name → ToolSpec mapping (snapshot under lock)."""
        with self._lock:
            return dict(self._specs_by_name)

    @property
    def categories(self) -> dict[str, list[str]]:
        """Return the current category → [tool_name, ...] mapping (snapshot under lock)."""
        with self._lock:
            return dict(self._categories)
