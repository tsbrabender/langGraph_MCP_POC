"""Pydantic schemas for the refresh_cache tool."""

from pydantic import BaseModel


class RefreshCacheInput(BaseModel):
    url: str
    ttl_seconds: int = 21600  # default 6 hours


class RefreshCacheOutput(BaseModel):
    url: str
    content: str
    content_length: int
    ttl_seconds: int
    previously_cached: bool
