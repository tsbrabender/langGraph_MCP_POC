"""get_topic_resources tool package — exposes get_tool() for dynamic discovery."""

from app.mcp_server.tool_spec import ToolSpec
from app.mcp_server.tools.context_retrieval.get_topic_resources.schemas import GetTopicResourcesInput
from app.mcp_server.tools.context_retrieval.get_topic_resources.tool import run


def get_tool() -> ToolSpec:
    """Return the ToolSpec for get_topic_resources."""
    return ToolSpec(
        name="get_topic_resources",
        category="context_retrieval",
        description=(
            "Return the list of configured resource URLs for a topic (e.g. 'dyslexia', 'adhd'). "
            "Use when the user asks about a topic that may have external reference material."
        ),
        input_schema_class=GetTopicResourcesInput,
        handler=run,
    )
