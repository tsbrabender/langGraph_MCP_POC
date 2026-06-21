"""Pydantic schemas for the fetch_web_resource tool."""

from pydantic import BaseModel


class FetchWebResourceInput(BaseModel):
    url: str


class FetchWebResourceOutput(BaseModel):
    url: str
    content: str
    content_length: int
    status_code: int
    truncated: bool
