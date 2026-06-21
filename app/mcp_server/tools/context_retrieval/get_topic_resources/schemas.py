"""Pydantic schemas for the get_topic_resources tool."""

from pydantic import BaseModel


class GetTopicResourcesInput(BaseModel):
    topic: str


class GetTopicResourcesOutput(BaseModel):
    topic: str
    urls: list[str]
    ttl_seconds: int
    found: bool
