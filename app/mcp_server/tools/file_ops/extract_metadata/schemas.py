"""Pydantic schemas for the extract_metadata tool."""

from pydantic import BaseModel


class ExtractMetadataInput(BaseModel):
    path: str


class ExtractMetadataOutput(BaseModel):
    path: str
    name: str
    extension: str
    size_bytes: int
    created_at: str   # ISO-8601 UTC
    modified_at: str  # ISO-8601 UTC
    is_file: bool
    is_dir: bool
    permissions: str  # octal string, e.g. "0o644"
