"""Pydantic schemas for the list_files tool."""

from pydantic import BaseModel, Field


class FileEntry(BaseModel):
    name: str
    is_dir: bool
    size_bytes: int


class ListFilesInput(BaseModel):
    directory: str = Field(default=".", description="Directory path relative to sandbox root")


class ListFilesOutput(BaseModel):
    entries: list[FileEntry]
    count: int
    directory: str
