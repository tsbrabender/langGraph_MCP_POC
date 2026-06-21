"""get_cached_resource tool package — exposes get_tool() for dynamic discovery."""

from app.mcp_server.tool_spec import ToolSpec
from app.mcp_server.tools.context_retrieval.get_cached_resource.schemas import GetCachedResourceInput
from app.mcp_server.tools.context_retrieval.get_cached_resource.tool import run


def get_tool() -> ToolSpec:
    """Return the ToolSpec for get_cached_resource."""
    return ToolSpec(
        name="get_cached_resource",
        category="context_retrieval",
        description=(
            "Return cached content for a URL if it exists and has not expired. "
            "Use before fetch_web_resource to avoid redundant network requests."
        ),
        input_schema_class=GetCachedResourceInput,
        handler=run,
    )
