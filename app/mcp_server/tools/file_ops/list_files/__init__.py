"""list_files tool package — exposes get_tool() for dynamic discovery."""

from app.mcp_server.tool_spec import ToolSpec
from app.mcp_server.tools.file_ops.list_files.schemas import ListFilesInput
from app.mcp_server.tools.file_ops.list_files.tool import run


def get_tool() -> ToolSpec:
    """Return the ToolSpec for list_files."""
    return ToolSpec(
        name="list_files",
        category="file_ops",
        description=(
            "List files and subdirectories at the given path within the sandbox. "
            "Use when the user wants to see what files exist in a directory."
        ),
        input_schema_class=ListFilesInput,
        handler=run,
    )
