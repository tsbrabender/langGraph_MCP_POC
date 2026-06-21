"""Cache client protocol and shared data model.

ResourceCacheEntry is the common data shape regardless of backend.
CacheClient is the structural interface that SQLiteCacheClient implements.
New backends (Redis, in-memory) implement the same three methods.
"""

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class ResourceCacheEntry:
    """A single cached resource.

    Attributes:
        url:          The URL that was fetched.
        content:      Normalized plain-text content.
        last_fetched: UTC datetime of the most recent fetch.
        ttl_seconds:  How many seconds the entry remains valid.
    """

    url: str
    content: str
    last_fetched: datetime
    ttl_seconds: int

    @property
    def age_seconds(self) -> int:
        """Elapsed seconds since last_fetched (always >= 0)."""
        delta = datetime.now(UTC) - self.last_fetched
        return max(0, int(delta.total_seconds()))

    def is_expired(self) -> bool:
        """True when the entry is older than its TTL."""
        return self.age_seconds >= self.ttl_seconds


class CacheClient:
    """Structural interface for cache backends.

    Implementations must provide async get / set / delete.
    This class is not instantiated directly.
    """

    async def get(self, url: str) -> ResourceCacheEntry | None:
        raise NotImplementedError

    async def set(self, url: str, content: str, ttl_seconds: int = 21600) -> None:
        raise NotImplementedError

    async def delete(self, url: str) -> None:
        raise NotImplementedError
