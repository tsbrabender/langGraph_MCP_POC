"""Unit tests for all LangGraph node functions and routing edge functions.

All external dependencies (ToolSelector, MCPExecutor, ResponseSynthesizer)
are mocked so tests run without Ollama, Redis, or a live file system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.graph.nodes import (
    classify_intent,
    finalize_response,
    ingest_user_input,
    llm_response_synthesis,
    llm_tool_selection,
    mcp_tool_invocation,
)
from app.graph.edges.routing import (
    after_intent_classification,
    after_tool_invocation,
    after_tool_selection,
)
from app.llm.tool_selector import ToolCall
from app.utils.errors import MCPToolError, ResponseSynthesisError, ToolSelectionError


# ---------------------------------------------------------------------------
# ingest_user_input
# ---------------------------------------------------------------------------


class TestIngestUserInput:
    @pytest.fixture
    def node(self):
        return ingest_user_input.make_node()

    async def test_strips_whitespace(self, node) -> None:
        result = await node({"user_input": "  list files  "})
        assert result["user_input"] == "list files"

    async def test_sets_request_id_in_metadata(self, node) -> None:
        result = await node({"user_input": "hello"})
        assert "request_id" in result["metadata"]

    async def test_sets_started_at_in_metadata(self, node) -> None:
        result = await node({"user_input": "hello"})
        assert "started_at" in result["metadata"]
        assert "T" in result["metadata"]["started_at"]

    async def test_clears_error_on_success(self, node) -> None:
        result = await node({"user_input": "hello", "error": "previous error"})
        assert result.get("error") is None

    async def test_empty_input_sets_error(self, node) -> None:
        result = await node({"user_input": "   "})
        assert result["error"] is not None
        assert "empty" in result["error"].lower()

    async def test_preserves_existing_metadata(self, node) -> None:
        result = await node({"user_input": "hi", "metadata": {"custom": "value"}})
        assert result["metadata"]["custom"] == "value"

    async def test_does_not_overwrite_existing_request_id(self, node) -> None:
        result = await node({"user_input": "hi", "metadata": {"request_id": "abc-123"}})
        assert result["metadata"]["request_id"] == "abc-123"


# ---------------------------------------------------------------------------
# llm_tool_selection
# ---------------------------------------------------------------------------


class TestLlmToolSelection:
    @pytest.fixture
    def mock_selector(self) -> AsyncMock:
        selector = AsyncMock()
        selector.select.return_value = ToolCall(
            tool_name="list_files",
            arguments={"directory": "."},
            reasoning="User wants to list files.",
        )
        return selector

    @pytest.fixture
    def node(self, mock_selector):
        return llm_tool_selection.make_node(mock_selector)

    async def test_sets_selected_tool(self, node) -> None:
        result = await node({"user_input": "list files"})
        assert result["selected_tool"] == "list_files"

    async def test_sets_tool_arguments(self, node) -> None:
        result = await node({"user_input": "list files"})
        assert result["tool_arguments"] == {"directory": "."}

    async def test_clears_error_on_success(self, node) -> None:
        result = await node({"user_input": "list files"})
        assert result["error"] is None

    async def test_passes_intent_as_context(self, mock_selector) -> None:
        node = llm_tool_selection.make_node(mock_selector)
        await node({"user_input": "show me stuff", "intent": "read_file"})
        _, kwargs = mock_selector.select.call_args
        context = kwargs.get("context") or mock_selector.select.call_args[0][1]
        assert context is not None
        assert "read_file" in str(context)

    async def test_tool_selection_error_sets_error(self, mock_selector) -> None:
        mock_selector.select.side_effect = ToolSelectionError("bad response")
        node = llm_tool_selection.make_node(mock_selector)
        result = await node({"user_input": "do something"})
        assert "bad response" in result["error"]
        assert "selected_tool" not in result

    async def test_no_intent_passes_no_context(self, mock_selector) -> None:
        node = llm_tool_selection.make_node(mock_selector)
        await node({"user_input": "list files"})
        call_args = mock_selector.select.call_args
        context = call_args[1].get("context") if call_args[1] else None
        assert context is None


# ---------------------------------------------------------------------------
# mcp_tool_invocation
# ---------------------------------------------------------------------------


class TestMcpToolInvocation:
    @pytest.fixture
    def mock_executor(self) -> AsyncMock:
        executor = AsyncMock()
        executor.execute.return_value = {"entries": [], "count": 0, "directory": "."}
        return executor

    @pytest.fixture
    def node(self, mock_executor):
        return mcp_tool_invocation.make_node(mock_executor)

    async def test_sets_tool_output(self, node) -> None:
        result = await node({"user_input": "x", "selected_tool": "list_files", "tool_arguments": {}})
        assert result["tool_output"] == {"entries": [], "count": 0, "directory": "."}

    async def test_clears_error_on_success(self, node) -> None:
        result = await node({"user_input": "x", "selected_tool": "list_files", "tool_arguments": {}})
        assert result["error"] is None

    async def test_executor_called_with_correct_args(self, mock_executor) -> None:
        node = mcp_tool_invocation.make_node(mock_executor)
        await node({
            "user_input": "x",
            "selected_tool": "read_file",
            "tool_arguments": {"path": "hello.txt"},
        })
        mock_executor.execute.assert_called_once_with("read_file", {"path": "hello.txt"})

    async def test_missing_selected_tool_sets_error(self, node) -> None:
        result = await node({"user_input": "x"})
        assert result["error"] is not None

    async def test_executor_exception_sets_error(self, mock_executor) -> None:
        mock_executor.execute.side_effect = MCPToolError("file not found")
        node = mcp_tool_invocation.make_node(mock_executor)
        result = await node({"user_input": "x", "selected_tool": "read_file", "tool_arguments": {"path": "missing.txt"}})
        assert "file not found" in result["error"]
        assert "tool_output" not in result

    async def test_uses_empty_dict_for_missing_tool_arguments(self, mock_executor) -> None:
        node = mcp_tool_invocation.make_node(mock_executor)
        await node({"user_input": "x", "selected_tool": "list_files"})
        mock_executor.execute.assert_called_once_with("list_files", {})


# ---------------------------------------------------------------------------
# llm_response_synthesis
# ---------------------------------------------------------------------------


class TestLlmResponseSynthesis:
    @pytest.fixture
    def mock_synthesizer(self) -> AsyncMock:
        synthesizer = AsyncMock()
        synthesizer.synthesize.return_value = "There are 3 files in the sandbox."
        return synthesizer

    @pytest.fixture
    def node(self, mock_synthesizer):
        return llm_response_synthesis.make_node(mock_synthesizer)

    async def test_sets_final_response(self, node) -> None:
        result = await node({
            "user_input": "list files",
            "selected_tool": "list_files",
            "tool_output": {"count": 3},
        })
        assert result["final_response"] == "There are 3 files in the sandbox."

    async def test_clears_error_on_success(self, node) -> None:
        result = await node({
            "user_input": "list files",
            "selected_tool": "list_files",
            "tool_output": {},
        })
        assert result["error"] is None

    async def test_error_in_state_skips_llm(self, mock_synthesizer) -> None:
        node = llm_response_synthesis.make_node(mock_synthesizer)
        result = await node({
            "user_input": "x",
            "error": "tool failed",
            "selected_tool": "list_files",
        })
        mock_synthesizer.synthesize.assert_not_called()
        assert "error" in result["final_response"].lower() or "tool failed" in result["final_response"]

    async def test_synthesis_error_sets_error(self, mock_synthesizer) -> None:
        mock_synthesizer.synthesize.side_effect = ResponseSynthesisError("empty response")
        node = llm_response_synthesis.make_node(mock_synthesizer)
        result = await node({
            "user_input": "x",
            "selected_tool": "list_files",
            "tool_output": {},
        })
        assert "empty response" in result["error"]


# ---------------------------------------------------------------------------
# finalize_response
# ---------------------------------------------------------------------------


class TestFinalizeResponse:
    @pytest.fixture
    def node(self):
        return finalize_response.make_node()

    async def test_appends_to_conversation_history(self, node) -> None:
        result = await node({
            "user_input": "list files",
            "final_response": "Done.",
            "conversation_history": [],
        })
        assert len(result["conversation_history"]) == 1
        assert result["conversation_history"][0]["user"] == "list files"
        assert result["conversation_history"][0]["response"] == "Done."

    async def test_preserves_existing_history(self, node) -> None:
        existing = [{"user": "previous", "tool": None, "response": "ok"}]
        result = await node({
            "user_input": "new request",
            "final_response": "New response.",
            "conversation_history": existing,
        })
        assert len(result["conversation_history"]) == 2

    async def test_sets_completed_at_in_metadata(self, node) -> None:
        result = await node({"user_input": "x", "final_response": "done", "metadata": {}})
        assert "completed_at" in result["metadata"]

    async def test_fallback_when_no_final_response(self, node) -> None:
        result = await node({"user_input": "x", "error": "something went wrong"})
        assert "error" in result["final_response"].lower() or "something went wrong" in result["final_response"]

    async def test_clears_error(self, node) -> None:
        result = await node({"user_input": "x", "final_response": "ok", "error": "old error"})
        assert result["error"] is None


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------


class TestClassifyIntent:
    @pytest.fixture
    def node(self):
        return classify_intent.make_node()

    async def test_list_files_simple(self, node) -> None:
        result = await node({"user_input": "list files"})
        assert result.get("intent") == "list_files"

    async def test_list_files_in_dir(self, node) -> None:
        result = await node({"user_input": "list files in subdir"})
        assert result.get("selected_tool") == "list_files"
        assert result["tool_arguments"]["directory"] == "subdir"

    async def test_read_file(self, node) -> None:
        result = await node({"user_input": "read notes.md"})
        assert result.get("selected_tool") == "read_file"
        assert result["tool_arguments"]["path"] == "notes.md"

    async def test_search_files(self, node) -> None:
        result = await node({"user_input": "find *.py"})
        assert result.get("selected_tool") == "search_files"
        assert result["tool_arguments"]["pattern"] == "*.py"

    async def test_summarize_file(self, node) -> None:
        result = await node({"user_input": "summarize report.txt"})
        assert result.get("selected_tool") == "summarize_file"
        assert result["tool_arguments"]["path"] == "report.txt"

    async def test_extract_metadata(self, node) -> None:
        result = await node({"user_input": "metadata for data.json"})
        assert result.get("selected_tool") == "extract_metadata"
        assert result["tool_arguments"]["path"] == "data.json"

    async def test_no_match_returns_none_intent(self, node) -> None:
        result = await node({"user_input": "what is the capital of France?"})
        assert result.get("intent") is None
        assert "selected_tool" not in result

    async def test_ls_command(self, node) -> None:
        result = await node({"user_input": "ls"})
        assert result.get("intent") == "list_files"


# ---------------------------------------------------------------------------
# Edge routing functions
# ---------------------------------------------------------------------------


class TestRoutingFunctions:
    def test_after_tool_selection_ok_routes_to_invocation(self) -> None:
        state = {"user_input": "x", "selected_tool": "list_files", "tool_arguments": {}}
        assert after_tool_selection(state) == "mcp_tool_invocation"

    def test_after_tool_selection_error_routes_to_finalize(self) -> None:
        state = {"user_input": "x", "error": "bad json"}
        assert after_tool_selection(state) == "finalize_response"

    def test_after_tool_invocation_always_routes_to_synthesis(self) -> None:
        assert after_tool_invocation({"user_input": "x", "error": "fail"}) == "llm_response_synthesis"
        assert after_tool_invocation({"user_input": "x", "tool_output": {}}) == "llm_response_synthesis"

    def test_after_intent_classification_with_selected_tool(self) -> None:
        state = {"user_input": "list files", "selected_tool": "list_files", "tool_arguments": {}}
        assert after_intent_classification(state) == "mcp_tool_invocation"

    def test_after_intent_classification_without_selected_tool(self) -> None:
        state = {"user_input": "something complex", "intent": "read_file"}
        assert after_intent_classification(state) == "llm_tool_selection"

    def test_after_intent_classification_no_match(self) -> None:
        state = {"user_input": "something complex"}
        assert after_intent_classification(state) == "llm_tool_selection"
