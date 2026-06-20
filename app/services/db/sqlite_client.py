"""Async SQLite client for persistent workflow history.

Connects once per process lifetime, applies all pending schema migrations on
connect, and exposes typed helpers for saving and querying workflow runs.

Usage:
    client = SQLiteClient()
    await client.connect()      # also runs migrations
    await client.save_run(run)
    runs = await client.list_runs(limit=50)
    await client.disconnect()

For in-memory testing:
    client = SQLiteClient(db_path=":memory:")
    await client.connect()
"""

import json
from pathlib import Path
from typing import Any

import aiosqlite

from app.services.db.models import WorkflowRun
from app.utils.config import get_settings
from app.utils.errors import DatabaseError
from app.utils.logging import get_logger

log = get_logger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class SQLiteClient:
    """Thin async wrapper around aiosqlite with migration support.

    Args:
        db_path: Path to the SQLite file, or ``":memory:"`` for tests.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or get_settings().sqlite_db_path
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the database connection, configure pragmas, and run migrations."""
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = await aiosqlite.connect(self._db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA foreign_keys=ON")
            await self._run_migrations()
            log.info("sqlite_connected", path=self._db_path)
        except Exception as exc:
            raise DatabaseError(
                f"Failed to connect to SQLite at {self._db_path!r}: {exc}"
            ) from exc

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            log.info("sqlite_disconnected")

    def _require_connected(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise DatabaseError("SQLiteClient is not connected. Call connect() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    async def _run_migrations(self) -> None:
        """Apply each *.sql file in the migrations directory in sorted order.

        Migrations use ``CREATE TABLE IF NOT EXISTS`` so they are idempotent.
        """
        conn = self._conn
        assert conn is not None
        for sql_file in sorted(_MIGRATIONS_DIR.glob("*.sql")):
            sql = sql_file.read_text(encoding="utf-8")
            for statement in (s.strip() for s in sql.split(";") if s.strip()):
                await conn.execute(statement)
        await conn.commit()
        log.info("sqlite_migrations_applied", dir=str(_MIGRATIONS_DIR))

    # ------------------------------------------------------------------
    # Generic helpers (used internally and in tests)
    # ------------------------------------------------------------------

    async def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a non-returning SQL statement (INSERT, UPDATE, DELETE, DDL).

        Args:
            sql:    Parameterized SQL string.
            params: Tuple of bound values.

        Raises:
            DatabaseError: If not connected or the query fails.
        """
        conn = self._require_connected()
        try:
            await conn.execute(sql, params)
            await conn.commit()
        except Exception as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Execute a SELECT and return all rows as plain dicts.

        Args:
            sql:    Parameterized SQL string.
            params: Tuple of bound values.

        Raises:
            DatabaseError: If not connected or the query fails.
        """
        conn = self._require_connected()
        try:
            async with conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:
            raise DatabaseError(f"Query failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Workflow run helpers
    # ------------------------------------------------------------------

    async def save_run(self, run: WorkflowRun) -> None:
        """Upsert a workflow run record.

        Uses INSERT OR REPLACE so the same request_id can be updated (e.g.,
        to stamp completed_at after synthesis finishes).

        Args:
            run: The WorkflowRun to persist.

        Raises:
            DatabaseError: If not connected or the write fails.
        """
        tool_output_json = (
            json.dumps(run.tool_output) if run.tool_output is not None else None
        )
        await self.execute(
            """
            INSERT OR REPLACE INTO workflow_runs
                (request_id, user_input, intent, selected_tool, tool_output,
                 final_response, error, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.request_id,
                run.user_input,
                run.intent,
                run.selected_tool,
                tool_output_json,
                run.final_response,
                run.error,
                run.started_at,
                run.completed_at,
            ),
        )
        log.info("sqlite_run_saved", request_id=run.request_id)

    async def get_run(self, request_id: str) -> WorkflowRun | None:
        """Retrieve a workflow run by request_id.

        Args:
            request_id: The UUID identifying the run.

        Returns:
            A WorkflowRun, or None if no matching record exists.
        """
        rows = await self.fetch_all(
            "SELECT * FROM workflow_runs WHERE request_id = ?",
            (request_id,),
        )
        if not rows:
            return None
        return _row_to_run(rows[0])

    async def list_runs(self, limit: int = 50) -> list[WorkflowRun]:
        """Return the most recent workflow runs, newest first.

        Args:
            limit: Maximum number of records to return (default 50).

        Returns:
            List of WorkflowRun objects.
        """
        rows = await self.fetch_all(
            "SELECT * FROM workflow_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_run(row) for row in rows]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SQLiteClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_run(row: dict[str, Any]) -> WorkflowRun:
    """Convert a raw DB row dict to a WorkflowRun, decoding JSON fields."""
    raw_output = row.get("tool_output")
    tool_output = json.loads(raw_output) if raw_output else None
    return WorkflowRun(
        request_id=row["request_id"],
        user_input=row["user_input"],
        intent=row.get("intent"),
        selected_tool=row.get("selected_tool"),
        tool_output=tool_output,
        final_response=row.get("final_response"),
        error=row.get("error"),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        created_at=row.get("created_at"),
    )
