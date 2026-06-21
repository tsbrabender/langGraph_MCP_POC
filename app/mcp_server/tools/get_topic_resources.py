"""MCP tool: get_topic_resources — maps a topic name to its configured URLs.

Returns the ordered list of resource URLs for a topic from topic_map.yaml.
Returns an empty list (not an error) when the topic has no configured resources.
"""

from pydantic import BaseModel

from app.utils.logging import get_logger
from app.utils.topic_config import load_topic_map

log = get_logger(__name__)


class GetTopicResourcesInput(BaseModel):
    topic: str


class GetTopicResourcesOutput(BaseModel):
    topic: str
    urls: list[str]
    ttl_seconds: int
    found: bool


async def run(topic: str) -> GetTopicResourcesOutput:
    """Return the resource URLs configured for a topic.

    Args:
        topic: Topic name (case-insensitive, e.g. "dyslexia").

    Returns:
        GetTopicResourcesOutput with urls and ttl_seconds populated when found.
    """
    log.info("get_topic_resources_start", topic=topic)

    topic_map = load_topic_map()
    entry = topic_map.get_entry(topic)

    if entry is None:
        log.info("get_topic_resources_not_found", topic=topic)
        return GetTopicResourcesOutput(topic=topic, urls=[], ttl_seconds=0, found=False)

    log.info("get_topic_resources_complete", topic=topic, url_count=len(entry.urls))
    return GetTopicResourcesOutput(
        topic=topic,
        urls=entry.urls,
        ttl_seconds=entry.ttl_seconds,
        found=True,
    )
