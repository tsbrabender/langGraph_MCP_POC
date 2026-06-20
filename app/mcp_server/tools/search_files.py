"""MCP tool: search_files — glob-pattern file search within the sandbox."""

from pathlib import Path

from pydantic import BaseModel, Field

from app.mcp_server.tools._sandbox import resolve_safe_path
from app.utils.errors import MCPToolError
from app.utils.logging import get_logger

log = get_logger(__name__)

MAX_RESULTS = 500


class SearchFilesInput(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '*.txt' or '**/*.py'")
    directory: str = Field(default=".", description="Directory to search, relative to sandbox root")


class SearchFilesOutput(BaseModel):
    matches: list[str]
    count: int
    pattern: str
    directory: str
    truncated: bool


async def run(pattern: str, directory: str, sandbox_root: Path) -> SearchFilesOutput:
    """Search for files matching a glob pattern within a sandboxed directory.

    Args:
        pattern: Glob pattern (e.g. '*.txt', '**/*.py').
        directory: Search root relative to sandbox_root (default ".").
        sandbox_root: Absolute path to the sandbox root directory.

    Returns:
        SearchFilesOutput with matching relative paths, sorted alphabetically.

    Raises:
        SandboxViolationError: If directory resolves outside the sandbox.
        MCPToolError: If the search directory does not exist.
    """
    safe_dir = resolve_safe_path(sandbox_root, directory)
    log.info("search_files", pattern=pattern, directory=directory)

    if not safe_dir.exists():
        raise MCPToolError(f"Search directory does not exist: '{directory}'")
    if not safe_dir.is_dir():
        raise MCPToolError(f"Search path is not a directory: '{directory}'")

    root_resolved = sandbox_root.resolve()
    matches: list[str] = []
    truncated = False

    for match in sorted(safe_dir.rglob(pattern)):
        # Double-check every match stays in sandbox (defensive; rglob should not escape).
        try:
            rel = str(match.resolve().relative_to(root_resolved))
            matches.append(rel)
        except ValueError:
            continue

        if len(matches) >= MAX_RESULTS:
            truncated = True
            break

    rel_dir = str(safe_dir.relative_to(root_resolved))
    return SearchFilesOutput(
        matches=matches,
        count=len(matches),
        pattern=pattern,
        directory=rel_dir,
        truncated=truncated,
    )
