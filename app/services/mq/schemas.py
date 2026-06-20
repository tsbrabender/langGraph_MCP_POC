"""Pydantic schemas for MQ request and response messages.

All messages are JSON-serialised before being pushed to Redis and
deserialised after being popped. Both ends validate through these models.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class RequestMessage(BaseModel):
    """Message published by the producer to the request queue."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)


class ResponseMessage(BaseModel):
    """Message published by the consumer to the per-request response key."""

    request_id: str
    final_response: str
    selected_tool: str | None = None
    tool_output: Any = None
    error: str | None = None
    completed_at: str = Field(default_factory=_now_iso)
