"""MCP tool: get_cached_resource — returns cached web content when still valid.

A cache miss (content missing or TTL expired) returns hit=False with content=None.
The caller should then invoke fetch_web_resource followed by a cache write.

Cache logic:
  - entry exists AND age < ttl_seconds  →  hit=True, return content
  - entry missing OR age >= ttl_seconds →  hit=False, content=None
"""

from typing import Any

from pydantic import BaseModel

from app.utils.logging import get_logger

log = get_logger(__name__)


class GetCachedResourceInput(BaseModel):
    url: str


class GetCachedResourceOutput(BaseModel):
    url: str
    content: str | None
    hit: bool
    age_seconds: int | None  # None on cache miss


async def run(url: str, cache_client: Any) -> GetCachedResourceOutput:
    """Check the cache for url and return content when valid.

    Args:
        url:          The URL to look up.
        cache_client: An injected SQLiteCacheClient (or compatible backend).

    Returns:
        GetCachedResourceOutput. Check .hit before using .content.
    """
    log.info("get_cached_resource_start", url=url)

    entry = await cache_client.get(url)

    if entry is None:
        log.info("get_cached_resource_miss", url=url, reason="not_found")
        return GetCachedResourceOutput(url=url, content=None, hit=False, age_seconds=None)

    if entry.is_expired():
        log.info("get_cached_resource_miss", url=url, reason="expired", age_seconds=entry.age_seconds)
        return GetCachedResourceOutput(url=url, content=None, hit=False, age_seconds=entry.age_seconds)

    log.info("get_cached_resource_hit", url=url, age_seconds=entry.age_seconds)
    return GetCachedResourceOutput(
        url=url,
        content=entry.content,
        hit=True,
        age_seconds=entry.age_seconds,
    )
