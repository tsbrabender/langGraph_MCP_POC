"""summarize_file tool package — exposes get_tool() for dynamic discovery."""

from app.mcp_server.tool_spec import ToolSpec
from app.mcp_server.tools.summarization.summarize_file.schemas import SummarizeFileInput
from app.mcp_server.tools.summarization.summarize_file.tool import run


def get_tool() -> ToolSpec:
    """Return the ToolSpec for summarize_file."""
    return ToolSpec(
        name="summarize_file",
        category="summarization",
        description=(
            "Summarize the content of a file using the local LLM. "
            "Use when the user wants a brief overview of a file without reading it in full."
        ),
        input_schema_class=SummarizeFileInput,
        handler=run,
    )
