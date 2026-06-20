"""MCP tool: list_files — enumerate entries in a sandboxed directory."""

from pathlib import Path

from pydantic import BaseModel, Field

from app.mcp_server.tools._sandbox import resolve_safe_path
from app.utils.errors import MCPToolError
from app.utils.logging import get_logger

log = get_logger(__name__)


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


async def run(directory: str, sandbox_root: Path) -> ListFilesOutput:
    """List files and subdirectories within a sandboxed directory.

    Args:
        directory: Path relative to sandbox_root (default ".").
        sandbox_root: Absolute path to the sandbox root directory.

    Returns:
        ListFilesOutput with sorted directory entries.

    Raises:
        SandboxViolationError: If directory resolves outside the sandbox.
        MCPToolError: If the path does not exist or is not a directory.
    """
    safe_path = resolve_safe_path(sandbox_root, directory)
    log.info("list_files", directory=directory, resolved=str(safe_path))

    if not safe_path.exists():
        raise MCPToolError(f"Directory does not exist: '{directory}'")
    if not safe_path.is_dir():
        raise MCPToolError(f"Path is not a directory: '{directory}'")

    entries: list[FileEntry] = []
    for item in sorted(safe_path.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        try:
            stat = item.stat()
            entries.append(FileEntry(name=item.name, is_dir=item.is_dir(), size_bytes=stat.st_size))
        except OSError:
            continue

    rel_dir = str(safe_path.relative_to(sandbox_root.resolve()))
    return ListFilesOutput(entries=entries, count=len(entries), directory=rel_dir)
