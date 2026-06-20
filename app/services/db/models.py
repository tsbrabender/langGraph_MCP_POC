"""Pydantic model representing a single workflow run row in SQLite."""

from typing import Any

from pydantic import BaseModel


class WorkflowRun(BaseModel):
    """One completed workflow execution stored in the workflow_runs table."""

    request_id: str
    user_input: str
    intent: str | None = None
    selected_tool: str | None = None
    tool_output: Any | None = None      # dict — stored as JSON in the DB
    final_response: str | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str | None = None       # Set by SQLite default on INSERT
