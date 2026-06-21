"""MCP tool: fetch_web_resource — retrieves and normalizes content from a URL.

Performs an async HTTP GET, extracts readable text from HTML (stripping scripts,
styles, navigation), collapses whitespace, and returns plain text capped at
FETCH_MAX_CHARS. For non-HTML responses the raw text is returned as-is.

The tool is deterministic for the same URL content — it does not cache.
Caching is the responsibility of the caller (see get_cached_resource / refresh_cache).
"""

import html
import re
from html.parser import HTMLParser

import httpx

from app.mcp_server.tools.context_retrieval.fetch_web_resource.schemas import FetchWebResourceOutput
from app.utils.errors import MCPToolError
from app.utils.logging import get_logger

log = get_logger(__name__)

FETCH_TIMEOUT_SECONDS = 15
FETCH_MAX_BYTES = 512 * 1024   # 512 KB — cap raw download
FETCH_MAX_CHARS = 8_000        # cap normalized text returned to the LLM

# Tags whose full subtree should be excluded from text extraction.
_SKIP_TAGS = frozenset({"script", "style", "head", "nav", "footer", "aside", "noscript"})


class _TextExtractor(HTMLParser):
    """Minimal stdlib HTML → plain-text extractor."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        joined = " ".join(self._parts)
        joined = html.unescape(joined)
        return re.sub(r"\s{2,}", " ", joined).strip()


def _extract_text(raw: str) -> str:
    """Return plain text from an HTML string, stripping all markup."""
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        return parser.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", raw).strip()


async def run(url: str) -> FetchWebResourceOutput:
    """Fetch url and return normalized plain text.

    Args:
        url: Fully-qualified URL to retrieve.

    Returns:
        FetchWebResourceOutput with content and metadata.

    Raises:
        MCPToolError: On HTTP error or network failure.
    """
    log.info("fetch_web_resource_start", url=url)

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=FETCH_TIMEOUT_SECONDS,
        ) as client:
            response = await client.get(
                url, headers={"User-Agent": "LangGraphMCPAgent/1.0 (educational)"}
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        log.error("fetch_web_resource_http_error", url=url, status=exc.response.status_code)
        raise MCPToolError(f"HTTP {exc.response.status_code} fetching {url}") from exc
    except httpx.RequestError as exc:
        log.error("fetch_web_resource_network_error", url=url, error=str(exc))
        raise MCPToolError(f"Network error fetching {url}: {exc}") from exc

    raw_bytes = response.content[:FETCH_MAX_BYTES]
    encoding = response.encoding or "utf-8"
    raw_text = raw_bytes.decode(encoding, errors="replace")

    content_type = response.headers.get("content-type", "")
    if "html" in content_type:
        content = _extract_text(raw_text)
    else:
        content = raw_text.strip()

    truncated = len(content) > FETCH_MAX_CHARS
    if truncated:
        content = content[:FETCH_MAX_CHARS]

    result = FetchWebResourceOutput(
        url=url,
        content=content,
        content_length=len(content),
        status_code=response.status_code,
        truncated=truncated,
    )
    log.info(
        "fetch_web_resource_complete",
        url=url,
        content_length=len(content),
        truncated=truncated,
    )
    return result
