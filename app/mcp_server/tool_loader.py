"""Dynamic MCP tool discovery and registration.

Recursively scans app/mcp_server/tools/ for category folders, imports each
tool's get_tool() factory, and returns structured ToolSpec objects.

Supports live reload: call discover_all_tools() at any time to pick up tools
added, removed, or moved without restarting the server.
"""

import importlib
import sys
from pathlib import Path

from app.mcp_server.tool_spec import ToolSpec
from app.utils.logging import get_logger

log = get_logger(__name__)

_TOOLS_DIR = Path(__file__).parent / "tools"


def discover_categories() -> list[Path]:
    """Return all category subdirectories in the tools directory.

    Excludes entries whose names start with '_' (shared utilities like _sandbox.py)
    and any non-directory entries.

    Returns:
        Sorted list of category directory Paths.
    """
    return sorted(
        p for p in _TOOLS_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )


def discover_tools(category_path: Path) -> list[ToolSpec]:
    """Import and return all ToolSpec objects from a category directory.

    Each tool subfolder must contain an __init__.py that exposes get_tool() -> ToolSpec.
    Folders whose names start with '_' are skipped.

    Args:
        category_path: Absolute path to a category folder (e.g. .../tools/file_ops).

    Returns:
        List of ToolSpec objects for successfully imported tools.
        Tools that fail to import are logged and skipped (never raise).
    """
    specs: list[ToolSpec] = []

    for tool_dir in sorted(category_path.iterdir()):
        if not tool_dir.is_dir() or tool_dir.name.startswith("_"):
            continue

        module_path = f"app.mcp_server.tools.{category_path.name}.{tool_dir.name}"

        # Evict from sys.modules so reimport picks up on-disk changes.
        if module_path in sys.modules:
            del sys.modules[module_path]

        try:
            module = importlib.import_module(module_path)
            spec: ToolSpec = module.get_tool()
            log.info(
                "tool_discovered",
                category=category_path.name,
                tool=spec.name,
                schema=spec.input_schema_class.__name__,
            )
            specs.append(spec)
        except Exception as exc:
            log.error(
                "tool_discovery_failed",
                module=module_path,
                error=str(exc),
                exc_info=True,
            )

    return specs


def discover_all_tools() -> dict[str, list[ToolSpec]]:
    """Discover all tools grouped by category.

    Returns:
        Mapping of category_name → list[ToolSpec]. Categories with zero
        successfully-loaded tools are excluded from the result.
    """
    result: dict[str, list[ToolSpec]] = {}

    for category_path in discover_categories():
        specs = discover_tools(category_path)
        if specs:
            result[category_path.name] = specs
            log.info(
                "category_loaded",
                category=category_path.name,
                tool_count=len(specs),
                tools=[s.name for s in specs],
            )

    return result
