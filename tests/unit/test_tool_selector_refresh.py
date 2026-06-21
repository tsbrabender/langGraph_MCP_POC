"""Unit tests verifying ToolSelector uses the live ToolRegistry on every call.

These tests use mock registries and mock LLM clients — no Ollama required.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.llm.tool_selector import ToolCall, ToolDefinition, ToolSelector
from app.utils.errors import ToolSelectionError


class _SimpleInput(BaseModel):
    directory: str = "."


def _mock_registry(*tool_names: str) -> MagicMock:
    """Return a mock ToolRegistry whose .definitions list contains the given names."""
    from app.llm.tool_selector import ToolDefinition

    defs = [
        ToolDefinition(name=n, description=f"Does {n}", input_schema_class=_SimpleInput)
        for n in tool_names
    ]
    reg = MagicMock()
    reg.definitions = defs
    return reg


def _llm_returning(tool_name: str, args: dict | None = None) -> AsyncMock:
    """Return a mock LLM that responds with a valid tool selection JSON."""
    llm = AsyncMock()
    llm.chat.return_value = json.dumps({
        "tool_name": tool_name,
        "arguments": args or {"directory": "."},
        "reasoning": "test selection",
    })
    return llm


# ---------------------------------------------------------------------------
# Basic selection
# ---------------------------------------------------------------------------


class TestToolSelectorBasic:
    async def test_selects_known_tool(self):
        registry = _mock_registry("list_files")
        selector = ToolSelector(_llm_returning("list_files"), registry)
        result = await selector.select("list files")
        assert result.tool_name == "list_files"
        assert isinstance(result, ToolCall)

    async def test_returns_validated_arguments(self):
        registry = _mock_registry("list_files")
        selector = ToolSelector(_llm_returning("list_files", {"directory": "subdir"}), registry)
        result = await selector.select("list subdir")
        assert result.arguments["directory"] == "subdir"

    async def test_returns_reasoning(self):
        registry = _mock_registry("list_files")
        selector = ToolSelector(_llm_returning("list_files"), registry)
        result = await selector.select("list files")
        assert result.reasoning == "test selection"


# ---------------------------------------------------------------------------
# Live registry updates
# ---------------------------------------------------------------------------


class TestToolSelectorSeesLiveRegistry:
    async def test_sees_tool_added_after_construction(self):
        """Selector should pick up new tools added to the registry after init."""
        registry = _mock_registry("list_files")
        llm = _llm_returning("new_tool", {"directory": "."})
        selector = ToolSelector(llm, registry)

        # Add a new tool to the registry (simulating registry.reload()).
        registry.definitions = [
            ToolDefinition(name="list_files", description="list", input_schema_class=_SimpleInput),
            ToolDefinition(name="new_tool", description="new", input_schema_class=_SimpleInput),
        ]

        result = await selector.select("use new tool")
        assert result.tool_name == "new_tool"

    async def test_rejects_tool_removed_after_reload(self):
        """After a tool is removed from the registry, the selector must reject it."""
        registry = _mock_registry("list_files", "old_tool")
        llm = _llm_returning("old_tool")
        selector = ToolSelector(llm, registry)

        # Remove old_tool from the registry.
        registry.definitions = [
            ToolDefinition(name="list_files", description="list", input_schema_class=_SimpleInput),
        ]

        with pytest.raises(ToolSelectionError, match="old_tool"):
            await selector.select("use old tool")

    async def test_empty_registry_raises_on_any_selection(self):
        registry = _mock_registry("list_files")
        llm = _llm_returning("list_files")
        selector = ToolSelector(llm, registry)

        registry.definitions = []

        with pytest.raises(ToolSelectionError):
            await selector.select("list files")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestToolSelectorErrors:
    async def test_bad_json_raises_tool_selection_error(self):
        registry = _mock_registry("list_files")
        llm = AsyncMock()
        llm.chat.return_value = "NOT VALID JSON {{{{"
        selector = ToolSelector(llm, registry)

        with pytest.raises(ToolSelectionError, match="non-JSON"):
            await selector.select("list files")

    async def test_unknown_tool_name_raises(self):
        registry = _mock_registry("list_files")
        llm = _llm_returning("does_not_exist")
        selector = ToolSelector(llm, registry)

        with pytest.raises(ToolSelectionError, match="does_not_exist"):
            await selector.select("do something")

    async def test_invalid_arguments_raise_tool_selection_error(self):
        class _StrictInput(BaseModel):
            required_field: str  # no default — must be supplied

        reg = MagicMock()
        reg.definitions = [
            ToolDefinition(name="strict_tool", description="strict", input_schema_class=_StrictInput)
        ]
        # LLM returns args missing required_field
        llm = AsyncMock()
        llm.chat.return_value = json.dumps({
            "tool_name": "strict_tool",
            "arguments": {},
            "reasoning": "test",
        })
        selector = ToolSelector(llm, reg)

        with pytest.raises(ToolSelectionError, match="failed validation"):
            await selector.select("use strict tool")

    async def test_llm_receives_updated_tool_schemas_after_reload(self):
        """After registry reload, the system prompt in build_messages includes new tools."""
        registry = _mock_registry("list_files")
        llm = AsyncMock()
        llm.chat.return_value = json.dumps({
            "tool_name": "list_files",
            "arguments": {"directory": "."},
            "reasoning": "test",
        })
        selector = ToolSelector(llm, registry)

        # Add another tool to the registry.
        registry.definitions = [
            ToolDefinition(name="list_files", description="list", input_schema_class=_SimpleInput),
            ToolDefinition(name="read_file", description="read", input_schema_class=_SimpleInput),
        ]

        await selector.select("list files")

        # Inspect the system prompt sent to the LLM.
        call_args = llm.chat.call_args
        messages = call_args[0][0]
        system_content = messages[0]["content"]

        assert "list_files" in system_content
        assert "read_file" in system_content
