"""search_files tool package — exposes get_tool() for dynamic discovery."""

from app.mcp_server.tool_spec import ToolSpec
from app.mcp_server.tools.file_ops.search_files.schemas import SearchFilesInput
from app.mcp_server.tools.file_ops.search_files.tool import run


def get_tool() -> ToolSpec:
    """Return the ToolSpec for search_files."""
    return ToolSpec(
        name="search_files",
        category="file_ops",
        description=(
            "Search for files matching a glob pattern within the sandbox. "
            "Supports recursive patterns like '**/*.py'. "
            "Use when the user wants to find files by name or extension."
        ),
        input_schema_class=SearchFilesInput,
        handler=run,
    )
