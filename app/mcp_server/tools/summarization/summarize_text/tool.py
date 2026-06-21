"""MCP tool: summarize_text — LLM-powered summarization of arbitrary text (non-deterministic)."""

from typing import Any

from app.mcp_server.tools.summarization.summarize_text.schemas import SummarizeTextOutput
from app.utils.errors import ResponseSynthesisError
from app.utils.logging import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a precise document summarizer. "
    "Summarize the provided text in 2-3 concise sentences. "
    "State only what is in the content — do not invent information."
)


async def run(text: str, max_chars: int = 4_000, llm: Any = None) -> SummarizeTextOutput:
    """Summarize arbitrary text using the local Ollama LLM.

    This tool is intentionally non-deterministic: the same input may produce
    slightly different summaries across calls. This is documented behavior.

    Args:
        text: The text content to summarize.
        max_chars: Maximum characters to send to the LLM (excess is truncated).
        llm: An OllamaClient instance (injected by the executor).

    Returns:
        SummarizeTextOutput with the LLM-generated summary.

    Raises:
        ResponseSynthesisError: If the LLM returns an empty response.
    """
    truncated = len(text) > max_chars
    snippet = text[:max_chars]
    if truncated:
        snippet += "\n\n[... content truncated for summarization ...]"

    log.info("summarize_text", input_length=len(text), truncated=truncated)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": snippet},
    ]

    summary = await llm.chat(messages)
    summary = summary.strip()

    if not summary:
        raise ResponseSynthesisError("LLM returned an empty summary for the provided text")

    return SummarizeTextOutput(summary=summary, truncated=truncated)
