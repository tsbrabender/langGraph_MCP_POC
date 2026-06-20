"""Node: ingest_user_input — validates and normalises the raw user request.

This node has no external dependencies.
Returns a partial state update with the cleaned input and initial metadata.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from app.graph.state import GraphState
from app.utils.logging import get_logger

log = get_logger(__name__)


async def _run(state: GraphState) -> dict[str, Any]:
    user_input = (state.get("user_input") or "").strip()
    log.info("node_ingest_start", raw_length=len(user_input))

    if not user_input:
        log.warning("node_ingest_empty_input")
        return {"error": "User input cannot be empty.", "user_input": ""}

    metadata: dict[str, Any] = dict(state.get("metadata") or {})
    metadata.setdefault("request_id", str(uuid.uuid4()))
    metadata["started_at"] = datetime.now(tz=timezone.utc).isoformat()

    log.info("node_ingest_complete", user_input=user_input[:80])
    return {"user_input": user_input, "metadata": metadata, "error": None}


def make_node():
    """Return the ingest_user_input node callable for LangGraph."""
    return _run
