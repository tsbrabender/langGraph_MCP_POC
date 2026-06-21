"""extract_metadata tool package — exposes get_tool() for dynamic discovery."""

from app.mcp_server.tool_spec import ToolSpec
from app.mcp_server.tools.file_ops.extract_metadata.schemas import ExtractMetadataInput
from app.mcp_server.tools.file_ops.extract_metadata.tool import run


def get_tool() -> ToolSpec:
    """Return the ToolSpec for extract_metadata."""
    return ToolSpec(
        name="extract_metadata",
        category="file_ops",
        description=(
            "Return file system metadata for a path: name, extension, size, "
            "timestamps, and whether it is a file or directory. "
            "Use when the user asks about file details, not file content."
        ),
        input_schema_class=ExtractMetadataInput,
        handler=run,
    )
