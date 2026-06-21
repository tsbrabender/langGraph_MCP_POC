"""Node: retrieve_context — fetches topic resources and populates context_documents.

Orchestrates the full context-retrieval pipeline without LLM involvement:

  1. get_topic_resources(topic)          → list of URLs + per-topic TTL
  2. For each URL:
       a. get_cached_resource(url)       → cache hit: use content
       b. [miss] fetch_web_resource(url) → live fetch
       c. [miss] cache_client.set()      → write to cache for next request

The fetched content is written to state["context_documents"], which the
response synthesizer picks up automatically.

On partial failure (one URL errors) the node logs the error and continues
with the remaining URLs — it never fails the whole pipeline for a single
unreachable resource.

Dependencies (injected via make_node):
    cache_client: SQLiteCacheClient (or compatible backend)
"""

from typing import Any

from app.graph.state import GraphState
from app.mcp_server.tools import fetch_web_resource, get_cached_resource, get_topic_resources
from app.utils.errors import MCPToolError
from app.utils.logging import get_logger

log = get_logger(__name__)


def make_node(cache_client: Any):
    """Return the retrieve_context node callable with a bound cache client.

    Args:
        cache_client: Connected cache backend (SQLiteCacheClient or compatible).
    """

    async def node(state: GraphState) -> dict[str, Any]:
        topic = state.get("topic")
        if not topic:
            log.info("node_retrieve_context_skip", reason="no_topic")
            return {"context_documents": [], "context_retrieved": False}

        log.info("node_retrieve_context_start", topic=topic)

        # Step 1: resolve URLs for this topic
        resources = await get_topic_resources.run(topic)
        if not resources.found:
            log.info("node_retrieve_context_no_resources", topic=topic)
            return {"context_documents": [], "context_retrieved": False}

        documents: list[dict[str, Any]] = []

        for url in resources.urls:
            try:
                # Step 2a: try cache first
                cached = await get_cached_resource.run(url, cache_client=cache_client)

                if cached.hit:
                    log.info("node_retrieve_context_cache_hit", url=url, age_seconds=cached.age_seconds)
                    documents.append({"url": url, "content": cached.content, "source": "cache"})
                    continue

                # Step 2b: cache miss — fetch live
                log.info(
                    "node_retrieve_context_fetch",
                    url=url,
                    reason="miss" if cached.age_seconds is None else "expired",
                )
                fetched = await fetch_web_resource.run(url)

                # Step 2c: write back to cache
                await cache_client.set(url, fetched.content, ttl_seconds=resources.ttl_seconds)

                documents.append({"url": url, "content": fetched.content, "source": "live"})

            except MCPToolError as exc:
                # One failed URL must not abort the whole retrieval pass.
                log.warning("node_retrieve_context_url_failed", url=url, error=str(exc))
                documents.append({"url": url, "content": None, "source": "error", "error": str(exc)})

        successful = [d for d in documents if d.get("content")]
        log.info(
            "node_retrieve_context_complete",
            topic=topic,
            total=len(documents),
            successful=len(successful),
        )
        return {
            "context_documents": documents,
            "context_retrieved": len(successful) > 0,
        }

    return node
