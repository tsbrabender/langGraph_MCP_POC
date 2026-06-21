"""Integration tests for the LangGraph workflows.

Tests the full graph wiring end-to-end:
  - Real MCPExecutor backed by a temporary sandbox (actual file I/O)
  - Mock ToolSelector and ResponseSynthesizer (no live Ollama needed)
  - Full ainvoke() round-trip through both graph variants

These tests verify that:
  1. State flows correctly between all nodes.
  2. The compiled graph produces a final_response.
  3. Hybrid routing shortcuts work as expected.
  4. Error conditions are handled gracefully.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from app.graph.graph import build_llm_graph, build_hybrid_graph
from app.llm.tool_registry import ToolRegistry
from app.llm.tool_selector import ToolCall
from app.services.mcp_executor import MCPExecutor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """Populate a temporary sandbox and return its root."""
    (tmp_path / "hello.txt").write_text("Hello, world!\n")
    (tmp_path / "data.json").write_text('{"answer": 42}')
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested\n")
    return tmp_path


@pytest.fixture
def executor(sandbox: Path) -> MCPExecutor:
    """Real MCPExecutor backed by the temporary sandbox with a loaded ToolRegistry."""
    registry = ToolRegistry()
    registry.reload()
    return MCPExecutor(sandbox_root=sandbox, tool_registry=registry)


@pytest.fixture
def mock_selector() -> AsyncMock:
    """Mock ToolSelector that always selects list_files with directory='.'."""
    sel = AsyncMock()
    sel.select.return_value = ToolCall(
        tool_name="list_files",
        arguments={"directory": "."},
        reasoning="Integration test selection.",
    )
    return sel


@pytest.fixture
def mock_synthesizer() -> AsyncMock:
    """Mock ResponseSynthesizer that returns a fixed string."""
    synth = AsyncMock()
    synth.synthesize.return_value = "I found 3 items in the sandbox."
    return synth


# ---------------------------------------------------------------------------
# LLM-driven graph
# ---------------------------------------------------------------------------


class TestLlmGraph:
    @pytest.fixture
    def graph(self, mock_selector, executor, mock_synthesizer):
        return build_llm_graph(mock_selector, executor, mock_synthesizer)

    async def test_full_round_trip_produces_final_response(self, graph) -> None:
        result = await graph.ainvoke({"user_input": "list the files"})
        assert result["final_response"] == "I found 3 items in the sandbox."

    async def test_tool_output_populated(self, graph) -> None:
        result = await graph.ainvoke({"user_input": "list the files"})
        assert result["tool_output"] is not None
        assert result["tool_output"]["count"] == 3  # hello.txt, data.json, subdir

    async def test_selected_tool_in_final_state(self, graph) -> None:
        result = await graph.ainvoke({"user_input": "list the files"})
        assert result["selected_tool"] == "list_files"

    async def test_conversation_history_updated(self, graph) -> None:
        result = await graph.ainvoke({"user_input": "list the files"})
        assert len(result["conversation_history"]) == 1
        entry = result["conversation_history"][0]
        assert entry["user"] == "list the files"
        assert entry["tool"] == "list_files"
        assert entry["response"] == "I found 3 items in the sandbox."

    async def test_metadata_has_timestamps(self, graph) -> None:
        result = await graph.ainvoke({"user_input": "list the files"})
        assert "started_at" in result["metadata"]
        assert "completed_at" in result["metadata"]

    async def test_input_whitespace_stripped(self, graph) -> None:
        result = await graph.ainvoke({"user_input": "   list the files   "})
        assert result["user_input"] == "list the files"

    async def test_selector_receives_user_input(self, graph, mock_selector) -> None:
        await graph.ainvoke({"user_input": "show me files"})
        call_user_input = mock_selector.select.call_args[0][0]
        assert call_user_input == "show me files"

    async def test_synthesizer_receives_tool_output(self, graph, mock_synthesizer) -> None:
        await graph.ainvoke({"user_input": "list files"})
        call_kwargs = mock_synthesizer.synthesize.call_args
        assert call_kwargs is not None
        tool_output = call_kwargs[1].get("tool_output") or call_kwargs[0][2]
        assert tool_output is not None

    async def test_error_on_tool_selection_still_produces_response(
        self, executor, mock_synthesizer
    ) -> None:
        failing_selector = AsyncMock()
        from app.utils.errors import ToolSelectionError
        failing_selector.select.side_effect = ToolSelectionError("bad LLM response")
        graph = build_llm_graph(failing_selector, executor, mock_synthesizer)
        result = await graph.ainvoke({"user_input": "do something"})
        assert result.get("final_response") is not None
        assert len(result.get("conversation_history", [])) == 1

    async def test_read_file_tool(self, mock_selector, executor, mock_synthesizer) -> None:
        mock_selector.select.return_value = ToolCall(
            tool_name="read_file",
            arguments={"path": "hello.txt"},
            reasoning="Reading a file.",
        )
        graph = build_llm_graph(mock_selector, executor, mock_synthesizer)
        result = await graph.ainvoke({"user_input": "read hello.txt"})
        assert result["selected_tool"] == "read_file"
        assert "Hello, world!" in result["tool_output"]["content"]

    async def test_search_files_tool(self, mock_selector, executor, mock_synthesizer) -> None:
        mock_selector.select.return_value = ToolCall(
            tool_name="search_files",
            arguments={"pattern": "**/*.txt", "directory": "."},
            reasoning="Searching for txt files.",
        )
        graph = build_llm_graph(mock_selector, executor, mock_synthesizer)
        result = await graph.ainvoke({"user_input": "find txt files"})
        assert result["tool_output"]["count"] >= 2  # hello.txt + subdir/nested.txt


# ---------------------------------------------------------------------------
# Hybrid graph
# ---------------------------------------------------------------------------


class TestHybridGraph:
    @pytest.fixture
    def graph(self, mock_selector, executor, mock_synthesizer):
        return build_hybrid_graph(mock_selector, executor, mock_synthesizer)

    async def test_simple_list_skips_llm_selection(
        self, executor, mock_synthesizer, mock_selector
    ) -> None:
        """'list files' should be handled by classify_intent, skipping LLM selection."""
        graph = build_hybrid_graph(mock_selector, executor, mock_synthesizer)
        result = await graph.ainvoke({"user_input": "list files"})
        # ToolSelector should NOT have been called because classify_intent extracted everything
        mock_selector.select.assert_not_called()
        assert result["final_response"] is not None

    async def test_read_file_shortcut(self, executor, mock_synthesizer, mock_selector) -> None:
        graph = build_hybrid_graph(mock_selector, executor, mock_synthesizer)
        result = await graph.ainvoke({"user_input": "read hello.txt"})
        mock_selector.select.assert_not_called()
        assert result["selected_tool"] == "read_file"
        assert "Hello, world!" in result["tool_output"]["content"]

    async def test_complex_query_falls_through_to_llm(
        self, executor, mock_synthesizer, mock_selector
    ) -> None:
        """An unrecognised query should fall through to llm_tool_selection."""
        graph = build_hybrid_graph(mock_selector, executor, mock_synthesizer)
        result = await graph.ainvoke({"user_input": "what can you tell me about this project?"})
        mock_selector.select.assert_called_once()
        assert result["final_response"] is not None

    async def test_intent_hint_sent_when_partial_match(
        self, executor, mock_synthesizer, mock_selector
    ) -> None:
        """A query matching intent keyword but not extractable should pass hint to LLM.

        "show files please" hits the intent-only keyword pattern (show files → list_files)
        but fails the full-extraction pattern (trailing 'please' prevents the match).
        classify_intent therefore sets only intent, not selected_tool, so the hybrid
        router falls through to llm_tool_selection.
        """
        graph = build_hybrid_graph(mock_selector, executor, mock_synthesizer)
        await graph.ainvoke({"user_input": "show files please"})
        # LLM should be called — classify_intent set intent only, not selected_tool
        mock_selector.select.assert_called_once()

    async def test_conversation_history_updated_in_hybrid_mode(self, graph) -> None:
        result = await graph.ainvoke({"user_input": "list files"})
        assert len(result["conversation_history"]) == 1

    async def test_hybrid_produces_final_response(self, graph) -> None:
        result = await graph.ainvoke({"user_input": "list files"})
        assert result.get("final_response") is not None

    async def test_find_pattern_shortcut(self, executor, mock_synthesizer, mock_selector) -> None:
        graph = build_hybrid_graph(mock_selector, executor, mock_synthesizer)
        result = await graph.ainvoke({"user_input": "find *.txt"})
        mock_selector.select.assert_not_called()
        assert result["selected_tool"] == "search_files"
        assert result["tool_output"]["count"] >= 1
