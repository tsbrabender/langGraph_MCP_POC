"""Node: detect_topic — keyword-based topic detection from user input.

Scans the user input against the keyword lists in topic_map.yaml and sets
state["topic"] when a match is found. This node has no external dependencies
and never calls an LLM.

The hybrid router uses state["topic"] to decide whether to branch into the
context-retrieval path before intent classification.

Output (always a partial state update):
  - topic: str | None — the matched topic name, or None when no match
"""

from typing import Any

from app.graph.state import GraphState
from app.utils.logging import get_logger
from app.utils.topic_config import load_topic_map

log = get_logger(__name__)


async def _run(state: GraphState) -> dict[str, Any]:
    user_input = state.get("user_input", "")
    log.info("node_detect_topic_start", user_input=user_input[:80])

    topic_map = load_topic_map()
    topic = topic_map.detect_topic(user_input)

    if topic:
        log.info("node_detect_topic_match", topic=topic)
    else:
        log.info("node_detect_topic_no_match")

    return {"topic": topic}


def make_node():
    """Return the detect_topic node callable for LangGraph."""
    return _run
