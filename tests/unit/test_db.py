"""Unit tests for SQLiteClient and WorkflowRun model.

Uses a temporary on-disk SQLite file (via tmp_path) per test class so tests
are isolated. The client runs real migrations against a real SQLite engine —
no mocking is needed since aiosqlite is a pure-Python async wrapper.
"""

import pytest
from pathlib import Path

from app.services.db.models import WorkflowRun
from app.services.db.sqlite_client import SQLiteClient
from app.utils.errors import DatabaseError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client(tmp_path: Path) -> SQLiteClient:
    """A connected SQLiteClient backed by a temporary database file."""
    c = SQLiteClient(db_path=str(tmp_path / "test.db"))
    await c.connect()
    yield c
    await c.disconnect()


def make_run(
    request_id: str = "req-001",
    user_input: str = "list files",
    selected_tool: str = "list_files",
    final_response: str = "Found 3 files.",
    error: str | None = None,
) -> WorkflowRun:
    return WorkflowRun(
        request_id=request_id,
        user_input=user_input,
        selected_tool=selected_tool,
        tool_output={"count": 3, "entries": [], "directory": "."},
        final_response=final_response,
        error=error,
        started_at="2026-06-20T10:00:00+00:00",
        completed_at="2026-06-20T10:00:01+00:00",
    )


# ---------------------------------------------------------------------------
# WorkflowRun schema
# ---------------------------------------------------------------------------


class TestWorkflowRun:
    def test_required_fields(self) -> None:
        run = WorkflowRun(request_id="x", user_input="hi")
        assert run.request_id == "x"
        assert run.user_input == "hi"

    def test_optional_fields_default_to_none(self) -> None:
        run = WorkflowRun(request_id="x", user_input="hi")
        assert run.selected_tool is None
        assert run.tool_output is None
        assert run.final_response is None
        assert run.error is None

    def test_tool_output_accepts_dict(self) -> None:
        run = WorkflowRun(request_id="x", user_input="hi", tool_output={"count": 5})
        assert run.tool_output["count"] == 5

    def test_serializes_to_dict(self) -> None:
        run = make_run()
        d = run.model_dump()
        assert d["request_id"] == "req-001"
        assert d["selected_tool"] == "list_files"


# ---------------------------------------------------------------------------
# SQLiteClient — connection
# ---------------------------------------------------------------------------


class TestSQLiteClientConnect:
    async def test_connect_creates_schema(self, client: SQLiteClient) -> None:
        rows = await client.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_runs'"
        )
        assert rows, "workflow_runs table should exist after connect()"

    async def test_connect_is_idempotent(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "idempotent.db")
        # First connect applies migrations
        async with SQLiteClient(db_path=db_path) as c1:
            await c1.save_run(make_run(request_id="before"))
        # Second connect must not fail or duplicate tables
        async with SQLiteClient(db_path=db_path) as c2:
            run = await c2.get_run("before")
        assert run is not None
        assert run.request_id == "before"

    async def test_not_connected_raises_on_execute(self) -> None:
        c = SQLiteClient(db_path=":memory:")
        with pytest.raises(DatabaseError, match="not connected"):
            await c.execute("SELECT 1")

    async def test_context_manager_connects_and_disconnects(self, tmp_path: Path) -> None:
        async with SQLiteClient(db_path=str(tmp_path / "ctx.db")) as c:
            assert c._conn is not None
        assert c._conn is None


# ---------------------------------------------------------------------------
# SQLiteClient — save_run
# ---------------------------------------------------------------------------


class TestSQLiteClientSaveRun:
    async def test_saves_run_and_retrievable(self, client: SQLiteClient) -> None:
        await client.save_run(make_run(request_id="save-1"))
        row = await client.get_run("save-1")
        assert row is not None
        assert row.request_id == "save-1"

    async def test_all_fields_persisted(self, client: SQLiteClient) -> None:
        run = make_run(request_id="full-run", user_input="search *.py", selected_tool="search_files")
        await client.save_run(run)
        retrieved = await client.get_run("full-run")
        assert retrieved is not None
        assert retrieved.user_input == "search *.py"
        assert retrieved.selected_tool == "search_files"
        assert retrieved.tool_output == {"count": 3, "entries": [], "directory": "."}
        assert retrieved.final_response == "Found 3 files."
        assert retrieved.started_at == "2026-06-20T10:00:00+00:00"
        assert retrieved.completed_at == "2026-06-20T10:00:01+00:00"

    async def test_upsert_updates_existing(self, client: SQLiteClient) -> None:
        await client.save_run(make_run(request_id="upsert-1", final_response="v1"))
        await client.save_run(make_run(request_id="upsert-1", final_response="v2"))
        row = await client.get_run("upsert-1")
        assert row is not None
        assert row.final_response == "v2"

    async def test_error_field_persisted(self, client: SQLiteClient) -> None:
        run = make_run(request_id="err-run", error="tool failed")
        await client.save_run(run)
        retrieved = await client.get_run("err-run")
        assert retrieved is not None
        assert retrieved.error == "tool failed"

    async def test_none_tool_output_stored_as_null(self, client: SQLiteClient) -> None:
        run = WorkflowRun(request_id="null-out", user_input="hi")
        await client.save_run(run)
        retrieved = await client.get_run("null-out")
        assert retrieved is not None
        assert retrieved.tool_output is None

    async def test_tool_output_roundtrips_json(self, client: SQLiteClient) -> None:
        nested = {"entries": [{"name": "a.txt", "size": 42}], "count": 1}
        run = WorkflowRun(request_id="json-out", user_input="list", tool_output=nested)
        await client.save_run(run)
        retrieved = await client.get_run("json-out")
        assert retrieved is not None
        assert retrieved.tool_output == nested


# ---------------------------------------------------------------------------
# SQLiteClient — get_run
# ---------------------------------------------------------------------------


class TestSQLiteClientGetRun:
    async def test_returns_none_for_missing_id(self, client: SQLiteClient) -> None:
        result = await client.get_run("does-not-exist")
        assert result is None

    async def test_returns_run_for_existing_id(self, client: SQLiteClient) -> None:
        await client.save_run(make_run(request_id="get-1"))
        result = await client.get_run("get-1")
        assert result is not None
        assert isinstance(result, WorkflowRun)

    async def test_does_not_return_other_runs(self, client: SQLiteClient) -> None:
        await client.save_run(make_run(request_id="x"))
        await client.save_run(make_run(request_id="y"))
        result = await client.get_run("x")
        assert result is not None
        assert result.request_id == "x"


# ---------------------------------------------------------------------------
# SQLiteClient — list_runs
# ---------------------------------------------------------------------------


class TestSQLiteClientListRuns:
    async def test_returns_empty_for_new_db(self, client: SQLiteClient) -> None:
        runs = await client.list_runs()
        assert runs == []

    async def test_returns_all_saved_runs(self, client: SQLiteClient) -> None:
        await client.save_run(make_run(request_id="a"))
        await client.save_run(make_run(request_id="b"))
        await client.save_run(make_run(request_id="c"))
        runs = await client.list_runs()
        assert len(runs) == 3

    async def test_respects_limit(self, client: SQLiteClient) -> None:
        for i in range(10):
            await client.save_run(make_run(request_id=f"run-{i:02d}"))
        runs = await client.list_runs(limit=3)
        assert len(runs) == 3

    async def test_returns_runs_as_workflow_run_objects(self, client: SQLiteClient) -> None:
        await client.save_run(make_run(request_id="typed"))
        runs = await client.list_runs()
        assert all(isinstance(r, WorkflowRun) for r in runs)

    async def test_created_at_is_set_by_sqlite(self, client: SQLiteClient) -> None:
        await client.save_run(make_run(request_id="ts-test"))
        run = (await client.list_runs())[0]
        assert run.created_at is not None


# ---------------------------------------------------------------------------
# finalize_response node — DB persistence
# ---------------------------------------------------------------------------


class TestFinalizeResponsePersistence:
    async def test_saves_run_when_db_client_provided(self, tmp_path: Path) -> None:
        async with SQLiteClient(db_path=str(tmp_path / "node.db")) as db:
            from app.graph.nodes import finalize_response

            node = finalize_response.make_node(db_client=db)
            state = {
                "user_input": "list files",
                "selected_tool": "list_files",
                "tool_output": {"count": 2},
                "final_response": "Found 2 files.",
                "metadata": {"request_id": "node-test", "started_at": "2026-01-01T00:00:00+00:00"},
                "conversation_history": [],
                "error": None,
            }
            await node(state)
            run = await db.get_run("node-test")
        assert run is not None
        assert run.request_id == "node-test"
        assert run.selected_tool == "list_files"
        assert run.final_response == "Found 2 files."

    async def test_skips_persistence_when_no_db_client(self) -> None:
        from app.graph.nodes import finalize_response

        node = finalize_response.make_node()  # db_client=None
        state = {
            "user_input": "list files",
            "final_response": "ok",
            "metadata": {"request_id": "no-db"},
            "conversation_history": [],
            "error": None,
        }
        result = await node(state)
        assert result["final_response"] == "ok"

    async def test_db_error_does_not_crash_node(self) -> None:
        from unittest.mock import AsyncMock
        from app.graph.nodes import finalize_response

        bad_db = AsyncMock()
        bad_db.save_run.side_effect = Exception("DB is down")

        node = finalize_response.make_node(db_client=bad_db)
        state = {
            "user_input": "list files",
            "final_response": "ok",
            "metadata": {"request_id": "err-db"},
            "conversation_history": [],
            "error": None,
        }
        result = await node(state)
        # Node must still return normally despite DB failure
        assert result["final_response"] == "ok"
        assert result["error"] is None
