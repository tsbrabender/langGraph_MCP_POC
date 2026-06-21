"""refresh_cache tool package — exposes get_tool() for dynamic discovery."""

from app.mcp_server.tool_spec import ToolSpec
from app.mcp_server.tools.context_retrieval.refresh_cache.schemas import RefreshCacheInput
from app.mcp_server.tools.context_retrieval.refresh_cache.tool import run


def get_tool() -> ToolSpec:
    """Return the ToolSpec for refresh_cache."""
    return ToolSpec(
        name="refresh_cache",
        category="context_retrieval",
        description=(
            "Force a fresh fetch of a URL and overwrite the cached content. "
            "Use when the user explicitly requests up-to-date or fresh information."
        ),
        input_schema_class=RefreshCacheInput,
        handler=run,
        dependencies={"fetcher": "fetch_web_resource"},
    )
