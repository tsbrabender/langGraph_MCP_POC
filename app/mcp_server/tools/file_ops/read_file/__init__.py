"""read_file tool package — exposes get_tool() for dynamic discovery."""

from app.mcp_server.tool_spec import ToolSpec
from app.mcp_server.tools.file_ops.read_file.schemas import ReadFileInput
from app.mcp_server.tools.file_ops.read_file.tool import run


def get_tool() -> ToolSpec:
    """Return the ToolSpec for read_file."""
    return ToolSpec(
        name="read_file",
        category="file_ops",
        description=(
            "Read and return the full text content of a specific file within the sandbox. "
            "Use when the user wants to see or inspect the contents of a named file."
        ),
        input_schema_class=ReadFileInput,
        handler=run,
    )
