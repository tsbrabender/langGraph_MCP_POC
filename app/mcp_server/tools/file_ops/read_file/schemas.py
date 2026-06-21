"""Pydantic schemas for the read_file tool."""

from pydantic import BaseModel


class ReadFileInput(BaseModel):
    path: str


class ReadFileOutput(BaseModel):
    content: str
    path: str
    size_bytes: int
    truncated: bool
