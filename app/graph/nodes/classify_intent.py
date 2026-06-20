"""Node: classify_intent — keyword-based intent classification for hybrid routing.

Attempts to determine the appropriate MCP tool and arguments from the user's
input using pattern matching. Used only in the hybrid graph.

If classification succeeds with extractable arguments:
  - Sets selected_tool and tool_arguments so the hybrid router can skip LLM selection.

If classification finds a tool name but cannot extract arguments:
  - Sets intent only; the router still calls llm_tool_selection (with the intent as hint).

If no pattern matches:
  - Returns intent=None; the router falls back to full LLM tool selection.

This node has no external dependencies.
"""

import re
from typing import Any

from app.graph.state import GraphState
from app.utils.logging import get_logger

log = get_logger(__name__)

# Each entry: (compiled pattern, tool_name, argument_extractor | None)
# argument_extractor(match) → dict of arguments, or None if extraction uncertain.
_PATTERNS: list[tuple[re.Pattern, str, Any]] = [
    # "list files in <dir>" / "ls <dir>"
    (
        re.compile(r"(?:list(?:\s+all)?\s*files?\s+in|ls\s+)\s*['\"]?([^\s'\"]+)['\"]?", re.I),
        "list_files",
        lambda m: {"directory": m.group(1)},
    ),
    # "list files" / "ls" / "show files" (no directory — use default ".")
    (
        re.compile(r"^(?:list(?:\s+all)?\s*files?|ls|show\s+files?)\s*$", re.I),
        "list_files",
        lambda m: {},
    ),
    # "read <file>" / "open <file>" / "cat <file>" / "show me <file>"
    (
        re.compile(r"(?:read|open|cat|show\s+me|display)\s+['\"]?([^\s'\"]+)['\"]?", re.I),
        "read_file",
        lambda m: {"path": m.group(1)},
    ),
    # "find <pattern>" / "search for <pattern>" / "look for <pattern>"
    (
        re.compile(r"(?:find|search(?:\s+for)?|look\s+for)\s+['\"]?([^\s'\"]+)['\"]?", re.I),
        "search_files",
        lambda m: {"pattern": m.group(1)},
    ),
    # "summarize <file>" / "summary of <file>"
    (
        re.compile(r"(?:summarize|summary(?:\s+of)?)\s+['\"]?([^\s'\"]+)['\"]?", re.I),
        "summarize_file",
        lambda m: {"path": m.group(1)},
    ),
    # "metadata for <file>" / "info about <file>" / "details of <file>"
    (
        re.compile(
            r"(?:metadata(?:\s+(?:for|of))?|info(?:rmation)?(?:\s+(?:about|of|for))?|"
            r"details?(?:\s+(?:of|for|about))?)\s+['\"]?([^\s'\"]+)['\"]?",
            re.I,
        ),
        "extract_metadata",
        lambda m: {"path": m.group(1)},
    ),
]

# Fallback: tool name only (no argument extraction) — still sets intent as a hint.
_INTENT_ONLY: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\blist\b|\bshow\s+files\b|\bls\b", re.I), "list_files"),
    (re.compile(r"\bread\b|\bopen\b|\bcat\b", re.I), "read_file"),
    (re.compile(r"\bsearch\b|\bfind\b|\blook\s+for\b", re.I), "search_files"),
    (re.compile(r"\bsummar(?:y|ize)\b", re.I), "summarize_file"),
    (re.compile(r"\bmetadata\b|\binfo\b|\bdetails\b", re.I), "extract_metadata"),
]


def _classify(user_input: str) -> tuple[str | None, dict[str, Any] | None]:
    """Attempt to classify user_input into a (tool_name, arguments) pair.

    Returns:
        (tool_name, arguments) if full extraction succeeded.
        (tool_name, None) if only tool name could be determined.
        (None, None) if no match found.
    """
    for pattern, tool_name, extractor in _PATTERNS:
        m = pattern.search(user_input)
        if m:
            return tool_name, extractor(m)

    for pattern, tool_name in _INTENT_ONLY:
        if pattern.search(user_input):
            return tool_name, None

    return None, None


async def _run(state: GraphState) -> dict[str, Any]:
    user_input = state.get("user_input", "")
    log.info("node_classify_intent_start", user_input=user_input[:80])

    tool_name, arguments = _classify(user_input)

    if tool_name and arguments is not None:
        # Full extraction — hybrid router can skip LLM selection entirely.
        log.info("node_classify_intent_full", tool_name=tool_name, arguments=arguments)
        return {
            "intent": tool_name,
            "selected_tool": tool_name,
            "tool_arguments": arguments,
        }

    if tool_name:
        # Tool name only — pass as hint to LLM selection.
        log.info("node_classify_intent_hint", tool_name=tool_name)
        return {"intent": tool_name}

    log.info("node_classify_intent_no_match")
    return {"intent": None}


def make_node():
    """Return the classify_intent node callable for LangGraph."""
    return _run
