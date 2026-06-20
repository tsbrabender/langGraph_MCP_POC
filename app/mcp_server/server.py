"""FastMCP server entry point — registers all MCP tools and starts the server.

Run with:
    python -m app.mcp_server.server
"""

from pathlib import Path

from fastmcp import FastMCP

from app.llm.ollama_client import OllamaClient
from app.mcp_server.tools import (
    extract_metadata as _extract_metadata,
    list_files as _list_files,
    read_file as _read_file,
    search_files as _search_files,
    summarize_file as _summarize_file,
)
from app.utils.config import get_settings
from app.utils.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger(__name__)

mcp = FastMCP(
    name="langgraph-mcp-poc",
    instructions=(
        "File-system agent tools with sandboxed access. "
        f"All paths are relative to the sandbox root: {settings.sandbox_root}"
    ),
)

_sandbox_root = Path(settings.sandbox_root).resolve()
_sandbox_root.mkdir(parents=True, exist_ok=True)

_llm = OllamaClient()


# ---------------------------------------------------------------------------
# Tool registrations — thin wrappers that inject sandbox_root and LLM client.
# The implementation logic lives in app/mcp_server/tools/*.py and is
# unit-tested independently of FastMCP.
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_files(directory: str = ".") -> dict:
    """List files and subdirectories at the given path within the sandbox.

    Args:
        directory: Path relative to the sandbox root (default: sandbox root itself).
    """
    result = await _list_files.run(directory, _sandbox_root)
    return result.model_dump()


@mcp.tool()
async def read_file(path: str) -> dict:
    """Read and return the text content of a file within the sandbox.

    Args:
        path: File path relative to the sandbox root.
    """
    result = await _read_file.run(path, _sandbox_root)
    return result.model_dump()


@mcp.tool()
async def search_files(pattern: str, directory: str = ".") -> dict:
    """Search for files matching a glob pattern within the sandbox.

    Args:
        pattern: Glob pattern, e.g. '*.txt' or '**/*.py'.
        directory: Directory to search, relative to the sandbox root (default: root).
    """
    result = await _search_files.run(pattern, directory, _sandbox_root)
    return result.model_dump()


@mcp.tool()
async def summarize_file(path: str) -> dict:
    """Summarize the content of a file using the local Ollama LLM.

    This tool is non-deterministic — the LLM may produce different summaries
    for the same file across calls.

    Args:
        path: File path relative to the sandbox root.
    """
    result = await _summarize_file.run(path, _sandbox_root, _llm)
    return result.model_dump()


@mcp.tool()
async def extract_metadata(path: str) -> dict:
    """Return file system metadata for a path within the sandbox.

    Args:
        path: File or directory path relative to the sandbox root.
    """
    result = await _extract_metadata.run(path, _sandbox_root)
    return result.model_dump()


if __name__ == "__main__":
    log.info(
        "starting_mcp_server",
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
        sandbox_root=str(_sandbox_root),
    )
    mcp.run(
        transport="streamable-http",
        host=settings.mcp_server_host,
        port=settings.mcp_server_port,
    )
