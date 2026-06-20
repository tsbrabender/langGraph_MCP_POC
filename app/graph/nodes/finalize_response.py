"""Node: finalize_response — appends to conversation history and stamps metadata.

This node always runs last. It also acts as the error-path terminus: if
final_response is not yet set (because synthesis was skipped), it generates
a fallback message from the error field.

Optional SQLite persistence: pass a SQLiteClient to make_node() and every
completed run will be saved to the workflow_runs table. Errors from the DB
write are logged but do not affect the returned state.
"""

from datetime import datetime, timezone
from typing import Any

from app.graph.state import GraphState
from app.utils.logging import get_logger

log = get_logger(__name__)


def make_node(db_client: Any | None = None):
    """Return the finalize_response node callable.

    Args:
        db_client: Optional SQLiteClient. When provided, every completed run
                   is persisted to the workflow_runs table.
    """

    async def _run(state: GraphState) -> dict[str, Any]:
        final_response = state.get("final_response")
        error = state.get("error")

        # Fallback: if routing skipped synthesis entirely, generate an error message here.
        if not final_response:
            final_response = (
                f"Sorry, I encountered an error: {error}" if error else "No response generated."
            )

        history: list[dict[str, Any]] = list(state.get("conversation_history") or [])
        history.append(
            {
                "user": state.get("user_input", ""),
                "tool": state.get("selected_tool"),
                "response": final_response,
            }
        )

        metadata: dict[str, Any] = dict(state.get("metadata") or {})
        metadata["completed_at"] = datetime.now(tz=timezone.utc).isoformat()

        log.info("node_finalize_complete", history_length=len(history))

        # --- Optional SQLite persistence ---
        if db_client is not None:
            try:
                from app.services.db.models import WorkflowRun

                run = WorkflowRun(
                    request_id=metadata.get("request_id", ""),
                    user_input=state.get("user_input", ""),
                    intent=state.get("intent"),
                    selected_tool=state.get("selected_tool"),
                    tool_output=state.get("tool_output"),
                    final_response=final_response,
                    error=error,
                    started_at=metadata.get("started_at"),
                    completed_at=metadata["completed_at"],
                )
                await db_client.save_run(run)
            except Exception as exc:
                log.warning("db_persist_failed", error=str(exc))

        return {
            "final_response": final_response,
            "conversation_history": history,
            "metadata": metadata,
            "error": None,
        }

    return _run
