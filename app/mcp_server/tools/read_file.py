"""MCP tool: read_file — return the text content of a sandboxed file."""

from pathlib import Path

from pydantic import BaseModel

from app.mcp_server.tools._sandbox import resolve_safe_path
from app.utils.errors import MCPToolError
from app.utils.logging import get_logger

log = get_logger(__name__)

# Maximum bytes to read — protects against accidentally reading huge binaries.
MAX_READ_BYTES = 1_000_000  # 1 MB


class ReadFileInput(BaseModel):
    path: str


class ReadFileOutput(BaseModel):
    content: str
    path: str
    size_bytes: int
    truncated: bool


async def run(path: str, sandbox_root: Path) -> ReadFileOutput:
    """Read and return the text content of a file within the sandbox.

    Args:
        path: File path relative to sandbox_root.
        sandbox_root: Absolute path to the sandbox root directory.

    Returns:
        ReadFileOutput with the file's text content.

    Raises:
        SandboxViolationError: If path resolves outside the sandbox.
        MCPToolError: If the path does not exist or is not a regular file.
    """
    safe_path = resolve_safe_path(sandbox_root, path)
    log.info("read_file", path=path, resolved=str(safe_path))

    if not safe_path.exists():
        raise MCPToolError(f"File does not exist: '{path}'")
    if not safe_path.is_file():
        raise MCPToolError(f"Path is not a file: '{path}'")

    size_bytes = safe_path.stat().st_size
    truncated = size_bytes > MAX_READ_BYTES

    raw = safe_path.read_bytes()[:MAX_READ_BYTES]
    content = raw.decode("utf-8", errors="replace")

    rel_path = str(safe_path.relative_to(sandbox_root.resolve()))
    return ReadFileOutput(content=content, path=rel_path, size_bytes=size_bytes, truncated=truncated)
