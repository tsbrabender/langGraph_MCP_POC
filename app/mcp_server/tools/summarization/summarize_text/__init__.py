"""summarize_text tool package — exposes get_tool() for dynamic discovery."""

from app.mcp_server.tool_spec import ToolSpec
from app.mcp_server.tools.summarization.summarize_text.schemas import SummarizeTextInput
from app.mcp_server.tools.summarization.summarize_text.tool import run


def get_tool() -> ToolSpec:
    """Return the ToolSpec for summarize_text."""
    return ToolSpec(
        name="summarize_text",
        category="summarization",
        description=(
            "Summarize arbitrary text content using the local LLM. "
            "Use when the user provides raw text and wants a brief overview, "
            "without reading from a file."
        ),
        input_schema_class=SummarizeTextInput,
        handler=run,
    )
