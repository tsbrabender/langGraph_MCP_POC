"""SQLite-backed resource cache.

Reuses the existing aiosqlite connection from SQLiteClient so both the
workflow history and the resource cache share a single database file and
connection. The resource_cache table is created by migration 002.

Usage:
    cache = SQLiteCacheClient(db_client)   # inject the connected SQLiteClient
    entry = await cache.get(url)
    await cache.set(url, content, ttl_seconds=3600)
"""

from datetime import UTC, datetime
from typing import Any

from app.services.cache.cache_client import CacheClient, ResourceCacheEntry
from app.utils.errors import DatabaseError
from app.utils.logging import get_logger

log = get_logger(__name__)

_ISO_FMT = "%Y-%m-%dT%H:%M:%S"


def _now_iso() -> str:
    return datetime.now(UTC).strftime(_ISO_FMT)


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, _ISO_FMT).replace(tzinfo=UTC)


class SQLiteCacheClient(CacheClient):
    """Resource cache backed by the project's SQLite database.

    Args:
        db_client: A connected SQLiteClient instance (injected).
    """

    def __init__(self, db_client: Any) -> None:
        self._db = db_client

    async def get(self, url: str) -> ResourceCacheEntry | None:
        """Return the cache entry for url, or None if not present.

        The caller must check entry.is_expired() before using the content.
        """
        rows = await self._db.fetch_all(
            "SELECT url, content, last_fetched, ttl_seconds FROM resource_cache WHERE url = ?",
            (url,),
        )
        if not rows:
            log.debug("cache_get_miss", url=url)
            return None
        row = rows[0]
        entry = ResourceCacheEntry(
            url=row["url"],
            content=row["content"],
            last_fetched=_parse_iso(row["last_fetched"]),
            ttl_seconds=row["ttl_seconds"],
        )
        log.debug("cache_get_found", url=url, age_seconds=entry.age_seconds, expired=entry.is_expired())
        return entry

    async def set(self, url: str, content: str, ttl_seconds: int = 21600) -> None:
        """Insert or replace a cache entry for url."""
        now = _now_iso()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO resource_cache (url, content, last_fetched, ttl_seconds)
            VALUES (?, ?, ?, ?)
            """,
            (url, content, now, ttl_seconds),
        )
        log.info("cache_set", url=url, ttl_seconds=ttl_seconds)

    async def delete(self, url: str) -> None:
        """Remove a cache entry. No-op if the entry does not exist."""
        await self._db.execute(
            "DELETE FROM resource_cache WHERE url = ?",
            (url,),
        )
        log.info("cache_delete", url=url)
