"""MCP tool: summarize_file — LLM-powered file summarization (non-deterministic)."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.mcp_server.tools._sandbox import resolve_safe_path
from app.utils.errors import MCPToolError, ResponseSynthesisError
from app.utils.logging import get_logger

log = get_logger(__name__)

# Characters sent to the LLM — keeps context within typical model limits.
MAX_CONTENT_CHARS = 4_000

SYSTEM_PROMPT = (
    "You are a precise technical document summarizer. "
    "Summarize the provided file content in 2-3 concise sentences. "
    "State only what is in the content — do not invent information."
)


class SummarizeFileInput(BaseModel):
    path: str


class SummarizeFileOutput(BaseModel):
    summary: str
    path: str
    truncated: bool


async def run(path: str, sandbox_root: Path, llm: Any) -> SummarizeFileOutput:
    """Summarize the content of a sandboxed file using the local Ollama LLM.

    This tool is intentionally non-deterministic: the same file may produce
    slightly different summaries across calls. This is documented behavior.

    Args:
        path: File path relative to sandbox_root.
        sandbox_root: Absolute path to the sandbox root directory.
        llm: An OllamaClient instance (injected by the server).

    Returns:
        SummarizeFileOutput with the LLM-generated summary.

    Raises:
        SandboxViolationError: If path resolves outside the sandbox.
        MCPToolError: If the file does not exist or is not readable.
        ResponseSynthesisError: If the LLM returns an empty response.
    """
    safe_path = resolve_safe_path(sandbox_root, path)
    log.info("summarize_file", path=path, resolved=str(safe_path))

    if not safe_path.exists():
        raise MCPToolError(f"File does not exist: '{path}'")
    if not safe_path.is_file():
        raise MCPToolError(f"Path is not a file: '{path}'")

    raw = safe_path.read_bytes()
    content = raw.decode("utf-8", errors="replace")
    truncated = len(content) > MAX_CONTENT_CHARS
    snippet = content[:MAX_CONTENT_CHARS]
    if truncated:
        snippet += "\n\n[... file truncated for summarization ...]"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"File: {path}\n\n{snippet}"},
    ]

    summary = await llm.chat(messages)
    summary = summary.strip()

    if not summary:
        raise ResponseSynthesisError(f"LLM returned an empty summary for '{path}'")

    rel_path = str(safe_path.relative_to(sandbox_root.resolve()))
    return SummarizeFileOutput(summary=summary, path=rel_path, truncated=truncated)
