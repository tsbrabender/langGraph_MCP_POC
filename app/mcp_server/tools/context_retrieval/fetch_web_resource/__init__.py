"""fetch_web_resource tool package — exposes get_tool() for dynamic discovery."""

from app.mcp_server.tool_spec import ToolSpec
from app.mcp_server.tools.context_retrieval.fetch_web_resource.schemas import FetchWebResourceInput
from app.mcp_server.tools.context_retrieval.fetch_web_resource.tool import run


def get_tool() -> ToolSpec:
    """Return the ToolSpec for fetch_web_resource."""
    return ToolSpec(
        name="fetch_web_resource",
        category="context_retrieval",
        description=(
            "Fetch a web page or document from a URL and return its normalized plain-text content. "
            "Use when external reference material must be retrieved from the internet."
        ),
        input_schema_class=FetchWebResourceInput,
        handler=run,
    )
