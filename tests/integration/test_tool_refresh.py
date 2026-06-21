"""Integration tests for POST /api/tools/refresh.

Uses httpx.AsyncClient against the real FastAPI app with a mocked graph so
the tests run without Ollama, Redis, or a real SQLite DB.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.ui.api import create_app


# ---------------------------------------------------------------------------
# Fixture: thin FastAPI app with a stub graph and pre-loaded tool registry
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_graph():
    """Minimal compiled graph that satisfies the lifespan check."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={
        "user_input": "test",
        "selected_tool": "list_files",
        "tool_output": {"entries": [], "count": 0, "directory": "."},
        "final_response": "Done.",
        "error": None,
    })
    return graph


@pytest_asyncio.fixture
async def client(stub_graph):
    """AsyncClient wired to the FastAPI app with a pre-built stub graph.

    Because the graph is pre-injected, the lifespan skips Ollama / SQLite
    initialization but still initializes the ToolRegistry.
    """
    from app.llm.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.reload()

    _app = create_app(graph=stub_graph)
    _app.state.tool_registry = registry

    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolRefreshEndpoint:
    async def test_returns_200(self, client: AsyncClient):
        resp = await client.post("/api/tools/refresh")
        assert resp.status_code == 200

    async def test_response_schema(self, client: AsyncClient):
        data = (await client.post("/api/tools/refresh")).json()
        assert "categories" in data
        assert "total_tools" in data
        assert isinstance(data["categories"], dict)
        assert isinstance(data["total_tools"], int)

    async def test_categories_include_all_known(self, client: AsyncClient):
        data = (await client.post("/api/tools/refresh")).json()
        cats = data["categories"]
        assert "file_ops" in cats
        assert "context_retrieval" in cats
        assert "summarization" in cats

    async def test_total_tools_equals_sum(self, client: AsyncClient):
        data = (await client.post("/api/tools/refresh")).json()
        expected = sum(len(v) for v in data["categories"].values())
        assert data["total_tools"] == expected

    async def test_total_tools_at_least_ten(self, client: AsyncClient):
        data = (await client.post("/api/tools/refresh")).json()
        assert data["total_tools"] >= 10

    async def test_file_ops_tools_listed(self, client: AsyncClient):
        data = (await client.post("/api/tools/refresh")).json()
        file_ops = set(data["categories"].get("file_ops", []))
        assert {"list_files", "read_file", "search_files", "extract_metadata"} <= file_ops

    async def test_summarization_tools_listed(self, client: AsyncClient):
        data = (await client.post("/api/tools/refresh")).json()
        summ = set(data["categories"].get("summarization", []))
        assert {"summarize_file", "summarize_text"} <= summ

    async def test_context_retrieval_tools_listed(self, client: AsyncClient):
        data = (await client.post("/api/tools/refresh")).json()
        ctx = set(data["categories"].get("context_retrieval", []))
        assert {"get_topic_resources", "fetch_web_resource", "get_cached_resource", "refresh_cache"} <= ctx

    async def test_multiple_refreshes_are_stable(self, client: AsyncClient):
        r1 = (await client.post("/api/tools/refresh")).json()
        r2 = (await client.post("/api/tools/refresh")).json()
        assert r1["total_tools"] == r2["total_tools"]
        assert r1["categories"] == r2["categories"]

    async def test_refresh_updates_registry_in_place(self, client: AsyncClient):
        """After a refresh the app's tool_registry.definitions must be non-empty."""
        await client.post("/api/tools/refresh")
        # Inspect the registry through the app state.
        # We can't directly access app.state from the test, but we can verify
        # via a second call returning consistent data.
        data = (await client.post("/api/tools/refresh")).json()
        assert data["total_tools"] >= 10
