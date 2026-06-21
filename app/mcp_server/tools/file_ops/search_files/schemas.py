"""Pydantic schemas for the search_files tool."""

from pydantic import BaseModel, Field


class SearchFilesInput(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '*.txt' or '**/*.py'")
    directory: str = Field(default=".", description="Directory to search, relative to sandbox root")


class SearchFilesOutput(BaseModel):
    matches: list[str]
    count: int
    pattern: str
    directory: str
    truncated: bool
