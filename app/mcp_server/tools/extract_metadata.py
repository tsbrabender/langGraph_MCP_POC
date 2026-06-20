"""MCP tool: extract_metadata — return file system metadata for a sandboxed path."""

import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from app.mcp_server.tools._sandbox import resolve_safe_path
from app.utils.errors import MCPToolError
from app.utils.logging import get_logger

log = get_logger(__name__)


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


async def run(path: str, sandbox_root: Path) -> ExtractMetadataOutput:
    """Extract file system metadata for a path within the sandbox.

    Args:
        path: Path relative to sandbox_root.
        sandbox_root: Absolute path to the sandbox root directory.

    Returns:
        ExtractMetadataOutput with name, size, timestamps, and type information.

    Raises:
        SandboxViolationError: If path resolves outside the sandbox.
        MCPToolError: If the path does not exist.
    """
    safe_path = resolve_safe_path(sandbox_root, path)
    log.info("extract_metadata", path=path, resolved=str(safe_path))

    if not safe_path.exists():
        raise MCPToolError(f"Path does not exist: '{path}'")

    stat = safe_path.stat()
    created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    permissions = oct(stat.st_mode & 0o777)

    rel_path = str(safe_path.relative_to(sandbox_root.resolve()))
    return ExtractMetadataOutput(
        path=rel_path,
        name=safe_path.name,
        extension=safe_path.suffix,
        size_bytes=stat.st_size,
        created_at=created_at,
        modified_at=modified_at,
        is_file=safe_path.is_file(),
        is_dir=safe_path.is_dir(),
        permissions=permissions,
    )
