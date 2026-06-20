"""Unit tests for the two-pass LLM workflow: tool selection and response synthesis.

All tests mock the OllamaClient — no live Ollama instance required.
"""

import json
import pytest
from unittest.mock import AsyncMock

from app.llm.tool_selector import ToolCall, ToolDefinition, ToolSelector
from app.llm.response_synthesizer import ResponseSynthesizer
from app.llm.tool_registry import build_tool_definitions
from app.mcp_server.tools.list_files import ListFilesInput
from app.mcp_server.tools.read_file import ReadFileInput
from app.mcp_server.tools.search_files import SearchFilesInput
from app.utils.errors import ToolSelectionError, ResponseSynthesisError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> AsyncMock:
    client = AsyncMock()
    client.chat.return_value = ""
    return client


@pytest.fixture
def tool_defs() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="list_files",
            description="List files in a directory.",
            input_schema_class=ListFilesInput,
        ),
        ToolDefinition(
            name="read_file",
            description="Read a file's content.",
            input_schema_class=ReadFileInput,
        ),
        ToolDefinition(
            name="search_files",
            description="Search for files by glob pattern.",
            input_schema_class=SearchFilesInput,
        ),
    ]


@pytest.fixture
def selector(mock_llm: AsyncMock, tool_defs: list[ToolDefinition]) -> ToolSelector:
    return ToolSelector(mock_llm, tool_defs)


@pytest.fixture
def synthesizer(mock_llm: AsyncMock) -> ResponseSynthesizer:
    return ResponseSynthesizer(mock_llm)


# ---------------------------------------------------------------------------
# ToolSelector — happy path
# ---------------------------------------------------------------------------


class TestToolSelectorHappyPath:
    async def test_returns_tool_call_with_correct_name(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "list_files",
            "arguments": {"directory": "."},
            "reasoning": "User wants to see files.",
        })
        result = await selector.select("list all files")
        assert isinstance(result, ToolCall)
        assert result.tool_name == "list_files"

    async def test_returns_validated_arguments(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "list_files",
            "arguments": {"directory": "subdir"},
            "reasoning": "Listing subdir.",
        })
        result = await selector.select("list subdir")
        assert result.arguments == {"directory": "subdir"}

    async def test_default_arguments_applied(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        # LLM omits optional 'directory' — Pydantic should apply the default "."
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "list_files",
            "arguments": {},
            "reasoning": "Default directory.",
        })
        result = await selector.select("list files")
        assert result.arguments["directory"] == "."

    async def test_reasoning_preserved(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "read_file",
            "arguments": {"path": "notes.md"},
            "reasoning": "User wants to read notes.",
        })
        result = await selector.select("read notes.md")
        assert result.reasoning == "User wants to read notes."

    async def test_search_files_with_both_args(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "search_files",
            "arguments": {"pattern": "**/*.py", "directory": "src"},
            "reasoning": "Finding Python files in src.",
        })
        result = await selector.select("find python files in src")
        assert result.tool_name == "search_files"
        assert result.arguments["pattern"] == "**/*.py"
        assert result.arguments["directory"] == "src"

    async def test_llm_called_with_format_schema(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "list_files",
            "arguments": {},
            "reasoning": "x",
        })
        await selector.select("list files")
        _, kwargs = mock_llm.chat.call_args
        assert "format" in kwargs
        assert kwargs["format"]["type"] == "object"

    async def test_context_included_in_user_message(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "list_files",
            "arguments": {},
            "reasoning": "x",
        })
        ctx = {"intent": "file_listing"}
        await selector.select("show me files", context=ctx)
        messages = mock_llm.chat.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "file_listing" in user_msg


# ---------------------------------------------------------------------------
# ToolSelector — prompt construction
# ---------------------------------------------------------------------------


class TestToolSelectorPrompt:
    def test_system_message_contains_all_tool_names(
        self, selector: ToolSelector
    ) -> None:
        messages = selector.build_messages("test input")
        system = messages[0]["content"]
        assert "list_files" in system
        assert "read_file" in system
        assert "search_files" in system

    def test_system_message_contains_tool_descriptions(
        self, selector: ToolSelector
    ) -> None:
        messages = selector.build_messages("test input")
        system = messages[0]["content"]
        assert "List files in a directory." in system
        assert "Read a file's content." in system

    def test_system_message_contains_json_schemas(
        self, selector: ToolSelector
    ) -> None:
        messages = selector.build_messages("test input")
        system = messages[0]["content"]
        # Each tool schema is rendered as JSON — check for a known field name
        assert '"directory"' in system
        assert '"path"' in system

    def test_system_message_contains_examples(self, selector: ToolSelector) -> None:
        messages = selector.build_messages("test input")
        system = messages[0]["content"]
        assert "list_files" in system
        assert "reasoning" in system

    def test_user_message_contains_request(self, selector: ToolSelector) -> None:
        messages = selector.build_messages("find all text files")
        user = messages[1]["content"]
        assert "find all text files" in user

    def test_two_messages_returned(self, selector: ToolSelector) -> None:
        messages = selector.build_messages("test")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# ToolSelector — error handling
# ---------------------------------------------------------------------------


class TestToolSelectorErrors:
    async def test_unknown_tool_raises(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "delete_everything",
            "arguments": {},
            "reasoning": "Chaos.",
        })
        with pytest.raises(ToolSelectionError, match="unknown tool"):
            await selector.select("do something dangerous")

    async def test_non_json_response_raises(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = "Sure, I'll list the files for you!"
        with pytest.raises(ToolSelectionError, match="non-JSON"):
            await selector.select("list files")

    async def test_invalid_arguments_raises(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        # 'path' is required for read_file but 'directory' is not a valid field
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "read_file",
            "arguments": {"not_a_field": 123},
            "reasoning": "Bad args.",
        })
        with pytest.raises(ToolSelectionError, match="failed validation"):
            await selector.select("read a file")

    async def test_missing_required_argument_raises(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        # read_file requires 'path' — omitting it should fail Pydantic validation
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "read_file",
            "arguments": {},
            "reasoning": "Missing path.",
        })
        with pytest.raises(ToolSelectionError, match="failed validation"):
            await selector.select("read the file")

    async def test_empty_json_object_raises_for_required_fields(
        self, selector: ToolSelector, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = json.dumps({
            "tool_name": "read_file",
            "arguments": {},
            "reasoning": "",
        })
        with pytest.raises(ToolSelectionError):
            await selector.select("read something")


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_returns_all_five_tools(self) -> None:
        defs = build_tool_definitions()
        names = {d.name for d in defs}
        assert names == {"list_files", "read_file", "search_files", "summarize_file", "extract_metadata"}

    def test_each_definition_has_description(self) -> None:
        for d in build_tool_definitions():
            assert d.description, f"{d.name} has an empty description"

    def test_each_schema_is_valid_dict(self) -> None:
        for d in build_tool_definitions():
            schema = d.json_schema()
            assert isinstance(schema, dict)
            assert "properties" in schema or "type" in schema

    def test_tool_definitions_are_unique(self) -> None:
        defs = build_tool_definitions()
        names = [d.name for d in defs]
        assert len(names) == len(set(names)), "Duplicate tool names in registry"


# ---------------------------------------------------------------------------
# ResponseSynthesizer — happy path
# ---------------------------------------------------------------------------


class TestResponseSynthesizerHappyPath:
    async def test_returns_llm_string(
        self, synthesizer: ResponseSynthesizer, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = "There are 3 files: hello.txt, data.json, notes.md."
        result = await synthesizer.synthesize(
            user_input="list the files",
            tool_name="list_files",
            tool_output={"entries": [], "count": 3, "directory": "."},
        )
        assert result == "There are 3 files: hello.txt, data.json, notes.md."

    async def test_whitespace_stripped(
        self, synthesizer: ResponseSynthesizer, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = "  \n  The file contains JSON.  \n  "
        result = await synthesizer.synthesize(
            user_input="read data.json",
            tool_name="read_file",
            tool_output={"content": "{}", "path": "data.json", "size_bytes": 2, "truncated": False},
        )
        assert result == "The file contains JSON."

    async def test_dict_tool_output_serialized_in_prompt(
        self, synthesizer: ResponseSynthesizer, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = "Found 2 matches."
        tool_output = {"matches": ["a.txt", "b.txt"], "count": 2}
        await synthesizer.synthesize("find txt files", "search_files", tool_output)
        messages = mock_llm.chat.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "a.txt" in user_msg
        assert "b.txt" in user_msg

    async def test_string_tool_output_not_double_serialized(
        self, synthesizer: ResponseSynthesizer, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = "Here is the summary."
        await synthesizer.synthesize("summarize it", "summarize_file", "A plain string summary.")
        messages = mock_llm.chat.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "A plain string summary." in user_msg

    async def test_state_context_included_in_prompt(
        self, synthesizer: ResponseSynthesizer, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = "Done."
        await synthesizer.synthesize(
            user_input="list files",
            tool_name="list_files",
            tool_output={},
            state_context={"intent": "exploration"},
        )
        messages = mock_llm.chat.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "exploration" in user_msg


# ---------------------------------------------------------------------------
# ResponseSynthesizer — prompt construction
# ---------------------------------------------------------------------------


class TestResponseSynthesizerPrompt:
    def test_two_messages_returned(self, synthesizer: ResponseSynthesizer) -> None:
        messages = synthesizer.build_messages("input", "list_files", {})
        assert len(messages) == 2

    def test_system_role_first(self, synthesizer: ResponseSynthesizer) -> None:
        messages = synthesizer.build_messages("input", "list_files", {})
        assert messages[0]["role"] == "system"

    def test_user_message_contains_request(self, synthesizer: ResponseSynthesizer) -> None:
        messages = synthesizer.build_messages("show me files please", "list_files", {})
        assert "show me files please" in messages[1]["content"]

    def test_user_message_contains_tool_name(self, synthesizer: ResponseSynthesizer) -> None:
        messages = synthesizer.build_messages("input", "extract_metadata", {})
        assert "extract_metadata" in messages[1]["content"]

    def test_system_prompt_enforces_grounding(self, synthesizer: ResponseSynthesizer) -> None:
        messages = synthesizer.build_messages("input", "list_files", {})
        system = messages[0]["content"]
        assert "invent" in system.lower() or "only" in system.lower()


# ---------------------------------------------------------------------------
# ResponseSynthesizer — error handling
# ---------------------------------------------------------------------------


class TestResponseSynthesizerErrors:
    async def test_empty_response_raises(
        self, synthesizer: ResponseSynthesizer, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = ""
        with pytest.raises(ResponseSynthesisError):
            await synthesizer.synthesize("list files", "list_files", {})

    async def test_whitespace_only_response_raises(
        self, synthesizer: ResponseSynthesizer, mock_llm: AsyncMock
    ) -> None:
        mock_llm.chat.return_value = "   \n\t  "
        with pytest.raises(ResponseSynthesisError):
            await synthesizer.synthesize("list files", "list_files", {})
