"""Unit tests for the FastAPI UI layer.

All tests use pre-injected mock graph / producer via create_app() so no
live Ollama, Redis, or MCP server is needed.
"""

import json
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock

from app.ui.api import create_app
from app.services.mq.schemas import ResponseMessage


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _graph_state(
    final_response: str = "I found 2 files in the sandbox.",
    tool: str = "list_files",
    error: str | None = None,
) -> dict:
    return {
        "user_input": "list files",
        "selected_tool": tool,
        "tool_output": {"count": 2, "entries": [], "directory": "."},
        "final_response": final_response,
        "conversation_history": [],
        "metadata": {"request_id": "stub-id"},
        "error": error,
    }


@pytest.fixture
def mock_graph() -> AsyncMock:
    g = AsyncMock()
    g.ainvoke.return_value = _graph_state()
    return g


@pytest.fixture
def mock_producer() -> AsyncMock:
    from app.services.mq.schemas import RequestMessage

    p = AsyncMock()
    p.publish_request.return_value = RequestMessage(
        request_id="mq-req-001", user_input="list files"
    )
    p.get_response.return_value = ResponseMessage(
        request_id="mq-req-001",
        final_response="MQ response: found 2 files.",
        selected_tool="list_files",
        tool_output={"count": 2},
    )
    p.disconnect = AsyncMock()
    return p


@pytest.fixture
def direct_app(mock_graph: AsyncMock) -> object:
    """FastAPI app in direct mode (no MQ)."""
    return create_app(graph=mock_graph)


@pytest.fixture
def mq_app(mock_graph: AsyncMock, mock_producer: AsyncMock) -> object:
    """FastAPI app in MQ mode."""
    app = create_app(graph=mock_graph, producer=mock_producer)
    app.state.mq_enabled = True
    return app


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    async def test_returns_200(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/health")
        assert resp.status_code == 200

    async def test_direct_mode_flag(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            data = (await client.get("/api/health")).json()
        assert data["mq_enabled"] is False
        assert data["mode"] == "direct"

    async def test_mq_mode_flag(self, mq_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=mq_app), base_url="http://test"
        ) as client:
            data = (await client.get("/api/health")).json()
        assert data["mq_enabled"] is True
        assert data["mode"] == "mq"

    async def test_status_is_ok(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            data = (await client.get("/api/health")).json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# GET / (static HTML)
# ---------------------------------------------------------------------------


class TestIndexEndpoint:
    async def test_returns_200(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert resp.status_code == 200

    async def test_returns_html(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert "text/html" in resp.headers["content-type"]

    async def test_html_contains_send_button(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert "Send" in resp.text

    async def test_html_contains_api_chat_reference(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            resp = await client.get("/")
        assert "/api/chat" in resp.text


# ---------------------------------------------------------------------------
# POST /api/chat — direct mode
# ---------------------------------------------------------------------------


class TestChatEndpointDirect:
    async def test_returns_200(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/chat", json={"user_input": "list files"})
        assert resp.status_code == 200

    async def test_final_response_present(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert data["final_response"] == "I found 2 files in the sandbox."

    async def test_mode_is_direct(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert data["mode"] == "direct"

    async def test_selected_tool_in_response(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert data["selected_tool"] == "list_files"

    async def test_tool_output_in_response(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert data["tool_output"]["count"] == 2

    async def test_request_id_is_uuid(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert len(data["request_id"]) == 36

    async def test_user_input_echoed(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            data = (
                await client.post("/api/chat", json={"user_input": "read hello.txt"})
            ).json()
        assert data["user_input"] == "read hello.txt"

    async def test_graph_invoked_with_user_input(
        self, direct_app, mock_graph: AsyncMock
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            await client.post("/api/chat", json={"user_input": "search *.py"})
        state_arg = mock_graph.ainvoke.call_args[0][0]
        assert state_arg["user_input"] == "search *.py"

    async def test_whitespace_input_stripped(
        self, direct_app, mock_graph: AsyncMock
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            await client.post("/api/chat", json={"user_input": "  list files  "})
        state_arg = mock_graph.ainvoke.call_args[0][0]
        assert state_arg["user_input"] == "list files"

    async def test_empty_input_returns_422(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/chat", json={"user_input": "   "})
        assert resp.status_code == 422

    async def test_graph_error_returns_500(self, mock_graph: AsyncMock) -> None:
        mock_graph.ainvoke.side_effect = RuntimeError("Ollama is down")
        app = create_app(graph=mock_graph)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/chat", json={"user_input": "list files"})
        assert resp.status_code == 500
        assert "Ollama is down" in resp.json()["detail"]

    async def test_error_field_surfaced_from_graph(self, mock_graph: AsyncMock) -> None:
        mock_graph.ainvoke.return_value = _graph_state(error="tool not found")
        app = create_app(graph=mock_graph)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert data["error"] == "tool not found"

    async def test_none_final_response_replaced_with_fallback(
        self, mock_graph: AsyncMock
    ) -> None:
        mock_graph.ainvoke.return_value = _graph_state(final_response=None)  # type: ignore[arg-type]
        app = create_app(graph=mock_graph)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert data["final_response"]  # Non-empty fallback


# ---------------------------------------------------------------------------
# POST /api/chat — MQ mode
# ---------------------------------------------------------------------------


class TestChatEndpointMQ:
    async def test_returns_200(self, mq_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=mq_app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/chat", json={"user_input": "list files"})
        assert resp.status_code == 200

    async def test_mode_is_mq(self, mq_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=mq_app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert data["mode"] == "mq"

    async def test_final_response_from_mq(self, mq_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=mq_app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert data["final_response"] == "MQ response: found 2 files."

    async def test_producer_publish_called(
        self, mq_app, mock_producer: AsyncMock
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=mq_app), base_url="http://test"
        ) as client:
            await client.post("/api/chat", json={"user_input": "list files"})
        mock_producer.publish_request.assert_called_once_with("list files")

    async def test_producer_get_response_called_with_request_id(
        self, mq_app, mock_producer: AsyncMock
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=mq_app), base_url="http://test"
        ) as client:
            await client.post("/api/chat", json={"user_input": "list files"})
        mock_producer.get_response.assert_called_once_with("mq-req-001", timeout=30.0)

    async def test_mq_timeout_returns_504(self, mock_producer: AsyncMock) -> None:
        mock_producer.get_response.return_value = None  # Timeout
        app = create_app(producer=mock_producer)
        app.state.mq_enabled = True
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/chat", json={"user_input": "list files"})
        assert resp.status_code == 504

    async def test_graph_not_called_in_mq_mode(
        self, mq_app, mock_graph: AsyncMock
    ) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=mq_app), base_url="http://test"
        ) as client:
            await client.post("/api/chat", json={"user_input": "list files"})
        mock_graph.ainvoke.assert_not_called()

    async def test_mq_response_selected_tool(self, mq_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=mq_app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert data["selected_tool"] == "list_files"

    async def test_mq_response_tool_output(self, mq_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=mq_app), base_url="http://test"
        ) as client:
            data = (await client.post("/api/chat", json={"user_input": "list files"})).json()
        assert data["tool_output"]["count"] == 2


# ---------------------------------------------------------------------------
# GET /api/history
# ---------------------------------------------------------------------------


class TestHistoryEndpoint:
    async def test_returns_empty_when_no_db(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            data = (await client.get("/api/history")).json()
        assert data["runs"] == []
        assert data["total"] == 0

    async def test_returns_runs_from_db(self, mock_graph: AsyncMock) -> None:
        from app.services.db.models import WorkflowRun

        mock_db = AsyncMock()
        mock_db.list_runs.return_value = [
            WorkflowRun(
                request_id="h-001",
                user_input="list files",
                selected_tool="list_files",
                final_response="Found 3 files.",
            )
        ]
        app = create_app(graph=mock_graph, db_client=mock_db)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            data = (await client.get("/api/history")).json()
        assert data["total"] == 1
        assert data["runs"][0]["request_id"] == "h-001"

    async def test_limit_query_param_passed_to_db(self, mock_graph: AsyncMock) -> None:
        mock_db = AsyncMock()
        mock_db.list_runs.return_value = []
        app = create_app(graph=mock_graph, db_client=mock_db)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/api/history?limit=5")
        mock_db.list_runs.assert_called_once_with(limit=5)

    async def test_default_limit_is_20(self, mock_graph: AsyncMock) -> None:
        mock_db = AsyncMock()
        mock_db.list_runs.return_value = []
        app = create_app(graph=mock_graph, db_client=mock_db)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/api/history")
        mock_db.list_runs.assert_called_once_with(limit=20)

    async def test_returns_200(self, direct_app) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=direct_app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/history")
        assert resp.status_code == 200

    async def test_db_error_returns_500(self, mock_graph: AsyncMock) -> None:
        mock_db = AsyncMock()
        mock_db.list_runs.side_effect = Exception("DB is gone")
        app = create_app(graph=mock_graph, db_client=mock_db)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/history")
        assert resp.status_code == 500
