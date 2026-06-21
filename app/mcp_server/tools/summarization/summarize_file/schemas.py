"""Pydantic schemas for the summarize_file tool."""

from pydantic import BaseModel


class SummarizeFileInput(BaseModel):
    path: str


class SummarizeFileOutput(BaseModel):
    summary: str
    path: str
    truncated: bool
