"""MCP tool: refresh_cache — forces a fresh fetch and writes the result to cache.

Always performs a network fetch regardless of whether a valid cache entry exists.
Use this when the user explicitly requests up-to-date content ("use fresh data",
"ignore cache", "refresh").

After the fetch succeeds the new content replaces whatever was in the cache.
The previous entry (if any) is overwritten atomically via INSERT OR REPLACE.
"""

from typing import Any, Callable, Awaitable

from pydantic import BaseModel

from app.utils.logging import get_logger

log = get_logger(__name__)


class RefreshCacheInput(BaseModel):
    url: str
    ttl_seconds: int = 21600  # default 6 hours


class RefreshCacheOutput(BaseModel):
    url: str
    content: str
    content_length: int
    ttl_seconds: int
    previously_cached: bool


async def run(
    url: str,
    ttl_seconds: int = 21600,
    cache_client: Any = None,
    fetcher: Callable[[str], Awaitable[Any]] = None,
) -> RefreshCacheOutput:
    """Fetch url from the network and overwrite the cache entry.

    Args:
        url:             The URL to fetch and cache.
        ttl_seconds:     TTL to store with the new entry.
        cache_client:    Cache backend (SQLiteCacheClient or compatible).
        fetcher:         Async callable (url) -> FetchWebResourceOutput.

    Returns:
        RefreshCacheOutput with the freshly fetched content.

    Raises:
        MCPToolError: If the fetch fails (propagated from fetch_web_resource).
    """
    log.info("refresh_cache_start", url=url, ttl_seconds=ttl_seconds)

    # Check whether anything existed before (for observability only — we always overwrite).
    existing = await cache_client.get(url)
    previously_cached = existing is not None

    fetch_result = await fetcher(url)
    content: str = fetch_result.content

    await cache_client.set(url, content, ttl_seconds=ttl_seconds)

    result = RefreshCacheOutput(
        url=url,
        content=content,
        content_length=len(content),
        ttl_seconds=ttl_seconds,
        previously_cached=previously_cached,
    )
    log.info(
        "refresh_cache_complete",
        url=url,
        content_length=len(content),
        previously_cached=previously_cached,
    )
    return result
