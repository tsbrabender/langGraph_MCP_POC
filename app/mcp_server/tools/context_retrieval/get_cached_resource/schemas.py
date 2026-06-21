"""Pydantic schemas for the get_cached_resource tool."""

from pydantic import BaseModel


class GetCachedResourceInput(BaseModel):
    url: str


class GetCachedResourceOutput(BaseModel):
    url: str
    content: str | None
    hit: bool
    age_seconds: int | None  # None on cache miss
