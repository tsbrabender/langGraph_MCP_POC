"""Unit tests for all MCP tool implementations and the sandbox enforcer.

Tests use pytest's tmp_path fixture so each test gets an isolated, clean sandbox.
The summarize_file and summarize_text tests mock the OllamaClient to avoid a
live Ollama dependency.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from app.mcp_server.tools._sandbox import resolve_safe_path
from app.mcp_server.tools.file_ops.list_files.tool import run as list_files_run
from app.mcp_server.tools.file_ops.read_file.tool import run as read_file_run, MAX_READ_BYTES
from app.mcp_server.tools.file_ops.search_files.tool import run as search_files_run
from app.mcp_server.tools.file_ops.extract_metadata.tool import run as extract_metadata_run
from app.mcp_server.tools.summarization.summarize_file.tool import run as summarize_file_run
from app.mcp_server.tools.summarization.summarize_text.tool import run as summarize_text_run
from app.utils.errors import SandboxViolationError, MCPToolError, ResponseSynthesisError


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """Populate a temporary sandbox with a known set of files and return its root."""
    (tmp_path / "hello.txt").write_text("Hello, world!\nLine two.\n")
    (tmp_path / "data.json").write_text('{"key": "value", "count": 42}')
    (tmp_path / "notes.md").write_text("# Notes\n\nSome notes here.\n")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("Nested content\n")
    (sub / "nested.py").write_text("print('hello')\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Sandbox enforcer
# ---------------------------------------------------------------------------


class TestSandbox:
    def test_valid_file_path(self, sandbox: Path) -> None:
        result = resolve_safe_path(sandbox, "hello.txt")
        assert result == (sandbox / "hello.txt").resolve()

    def test_valid_nested_path(self, sandbox: Path) -> None:
        result = resolve_safe_path(sandbox, "subdir/nested.txt")
        assert result == (sandbox / "subdir" / "nested.txt").resolve()

    def test_dot_path_resolves_to_root(self, sandbox: Path) -> None:
        result = resolve_safe_path(sandbox, ".")
        assert result == sandbox.resolve()

    def test_path_traversal_blocked(self, sandbox: Path) -> None:
        with pytest.raises(SandboxViolationError):
            resolve_safe_path(sandbox, "../outside.txt")

    def test_deep_traversal_blocked(self, sandbox: Path) -> None:
        with pytest.raises(SandboxViolationError):
            resolve_safe_path(sandbox, "subdir/../../outside.txt")

    def test_absolute_path_inside_sandbox_allowed(self, sandbox: Path) -> None:
        inside = str(sandbox / "hello.txt")
        result = resolve_safe_path(sandbox, inside)
        assert result == (sandbox / "hello.txt").resolve()

    def test_absolute_path_outside_sandbox_blocked(self, sandbox: Path) -> None:
        with pytest.raises(SandboxViolationError):
            resolve_safe_path(sandbox, "/etc/passwd")


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


class TestListFiles:
    async def test_list_root_returns_all_entries(self, sandbox: Path) -> None:
        result = await list_files_run(".", sandbox)
        names = [e.name for e in result.entries]
        assert "hello.txt" in names
        assert "data.json" in names
        assert "notes.md" in names
        assert "subdir" in names
        assert result.count == 4

    async def test_list_subdir(self, sandbox: Path) -> None:
        result = await list_files_run("subdir", sandbox)
        names = [e.name for e in result.entries]
        assert "nested.txt" in names
        assert "nested.py" in names
        assert result.count == 2

    async def test_dirs_sorted_before_files(self, sandbox: Path) -> None:
        result = await list_files_run(".", sandbox)
        first = result.entries[0]
        assert first.is_dir, "Directories should be listed before files"

    async def test_entry_reports_size(self, sandbox: Path) -> None:
        result = await list_files_run(".", sandbox)
        txt = next(e for e in result.entries if e.name == "hello.txt")
        assert txt.size_bytes > 0

    async def test_nonexistent_directory_raises(self, sandbox: Path) -> None:
        with pytest.raises(MCPToolError, match="does not exist"):
            await list_files_run("nonexistent", sandbox)

    async def test_file_path_raises(self, sandbox: Path) -> None:
        with pytest.raises(MCPToolError, match="not a directory"):
            await list_files_run("hello.txt", sandbox)

    async def test_sandbox_violation_raises(self, sandbox: Path) -> None:
        with pytest.raises(SandboxViolationError):
            await list_files_run("../outside", sandbox)

    async def test_directory_field_is_relative(self, sandbox: Path) -> None:
        result = await list_files_run(".", sandbox)
        assert not result.directory.startswith("/")
        assert not result.directory.startswith("C:")


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    async def test_reads_text_content(self, sandbox: Path) -> None:
        result = await read_file_run("hello.txt", sandbox)
        assert "Hello, world!" in result.content

    async def test_returns_correct_size(self, sandbox: Path) -> None:
        result = await read_file_run("hello.txt", sandbox)
        assert result.size_bytes > 0

    async def test_path_is_relative(self, sandbox: Path) -> None:
        result = await read_file_run("hello.txt", sandbox)
        assert result.path == "hello.txt"

    async def test_nested_file(self, sandbox: Path) -> None:
        result = await read_file_run("subdir/nested.txt", sandbox)
        assert "Nested content" in result.content

    async def test_json_file_content(self, sandbox: Path) -> None:
        result = await read_file_run("data.json", sandbox)
        assert '"key"' in result.content
        assert result.truncated is False

    async def test_nonexistent_file_raises(self, sandbox: Path) -> None:
        with pytest.raises(MCPToolError, match="does not exist"):
            await read_file_run("missing.txt", sandbox)

    async def test_directory_path_raises(self, sandbox: Path) -> None:
        with pytest.raises(MCPToolError, match="not a file"):
            await read_file_run("subdir", sandbox)

    async def test_sandbox_violation_raises(self, sandbox: Path) -> None:
        with pytest.raises(SandboxViolationError):
            await read_file_run("../../secret.txt", sandbox)

    async def test_large_file_truncated(self, sandbox: Path) -> None:
        big_file = sandbox / "big.txt"
        big_file.write_bytes(b"x" * 1_100_000)
        result = await read_file_run("big.txt", sandbox)
        assert result.truncated is True
        assert len(result.content) <= MAX_READ_BYTES + 10


# ---------------------------------------------------------------------------
# search_files
# ---------------------------------------------------------------------------


class TestSearchFiles:
    async def test_find_txt_files(self, sandbox: Path) -> None:
        result = await search_files_run("*.txt", ".", sandbox)
        assert any(m.endswith("hello.txt") for m in result.matches)

    async def test_recursive_glob(self, sandbox: Path) -> None:
        result = await search_files_run("**/*.txt", ".", sandbox)
        file_names = [Path(m).name for m in result.matches]
        assert "hello.txt" in file_names
        assert "nested.txt" in file_names

    async def test_pattern_with_no_matches(self, sandbox: Path) -> None:
        result = await search_files_run("*.xyz", ".", sandbox)
        assert result.count == 0
        assert result.matches == []

    async def test_search_in_subdir(self, sandbox: Path) -> None:
        result = await search_files_run("*.py", "subdir", sandbox)
        assert result.count == 1
        assert any("nested.py" in m for m in result.matches)

    async def test_count_matches_len_matches(self, sandbox: Path) -> None:
        result = await search_files_run("**/*", ".", sandbox)
        assert result.count == len(result.matches)

    async def test_nonexistent_directory_raises(self, sandbox: Path) -> None:
        with pytest.raises(MCPToolError, match="does not exist"):
            await search_files_run("*.txt", "nonexistent", sandbox)

    async def test_sandbox_violation_raises(self, sandbox: Path) -> None:
        with pytest.raises(SandboxViolationError):
            await search_files_run("*.txt", "../outside", sandbox)

    async def test_matches_are_relative_paths(self, sandbox: Path) -> None:
        result = await search_files_run("*.txt", ".", sandbox)
        for m in result.matches:
            assert not m.startswith("/")
            assert not m.startswith("C:")


# ---------------------------------------------------------------------------
# extract_metadata
# ---------------------------------------------------------------------------


class TestExtractMetadata:
    async def test_file_metadata(self, sandbox: Path) -> None:
        result = await extract_metadata_run("hello.txt", sandbox)
        assert result.name == "hello.txt"
        assert result.extension == ".txt"
        assert result.is_file is True
        assert result.is_dir is False
        assert result.size_bytes > 0

    async def test_directory_metadata(self, sandbox: Path) -> None:
        result = await extract_metadata_run("subdir", sandbox)
        assert result.name == "subdir"
        assert result.is_dir is True
        assert result.is_file is False

    async def test_json_extension(self, sandbox: Path) -> None:
        result = await extract_metadata_run("data.json", sandbox)
        assert result.extension == ".json"

    async def test_timestamps_are_iso8601(self, sandbox: Path) -> None:
        result = await extract_metadata_run("hello.txt", sandbox)
        assert "T" in result.created_at
        assert "T" in result.modified_at

    async def test_path_is_relative(self, sandbox: Path) -> None:
        result = await extract_metadata_run("hello.txt", sandbox)
        assert result.path == "hello.txt"

    async def test_nonexistent_path_raises(self, sandbox: Path) -> None:
        with pytest.raises(MCPToolError, match="does not exist"):
            await extract_metadata_run("missing.txt", sandbox)

    async def test_sandbox_violation_raises(self, sandbox: Path) -> None:
        with pytest.raises(SandboxViolationError):
            await extract_metadata_run("../etc/passwd", sandbox)

    async def test_permissions_is_octal_string(self, sandbox: Path) -> None:
        result = await extract_metadata_run("hello.txt", sandbox)
        assert result.permissions.startswith("0o")


# ---------------------------------------------------------------------------
# summarize_file
# ---------------------------------------------------------------------------


class TestSummarizeFile:
    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        client = AsyncMock()
        client.chat.return_value = "This file contains a greeting and a second line."
        return client

    async def test_returns_llm_summary(self, sandbox: Path, mock_llm: AsyncMock) -> None:
        result = await summarize_file_run("hello.txt", sandbox, mock_llm)
        assert result.summary == "This file contains a greeting and a second line."

    async def test_path_is_relative(self, sandbox: Path, mock_llm: AsyncMock) -> None:
        result = await summarize_file_run("hello.txt", sandbox, mock_llm)
        assert result.path == "hello.txt"

    async def test_small_file_not_truncated(self, sandbox: Path, mock_llm: AsyncMock) -> None:
        result = await summarize_file_run("hello.txt", sandbox, mock_llm)
        assert result.truncated is False

    async def test_large_file_flagged_as_truncated(self, sandbox: Path, mock_llm: AsyncMock) -> None:
        big = sandbox / "big.txt"
        big.write_text("word " * 2_000)
        result = await summarize_file_run("big.txt", sandbox, mock_llm)
        assert result.truncated is True

    async def test_llm_called_with_messages(self, sandbox: Path, mock_llm: AsyncMock) -> None:
        await summarize_file_run("hello.txt", sandbox, mock_llm)
        mock_llm.chat.assert_called_once()
        messages = mock_llm.chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "hello.txt" in messages[1]["content"]

    async def test_empty_llm_response_raises(self, sandbox: Path, mock_llm: AsyncMock) -> None:
        mock_llm.chat.return_value = "   "
        with pytest.raises(ResponseSynthesisError):
            await summarize_file_run("hello.txt", sandbox, mock_llm)

    async def test_nonexistent_file_raises(self, sandbox: Path, mock_llm: AsyncMock) -> None:
        with pytest.raises(MCPToolError, match="does not exist"):
            await summarize_file_run("missing.txt", sandbox, mock_llm)

    async def test_directory_raises(self, sandbox: Path, mock_llm: AsyncMock) -> None:
        with pytest.raises(MCPToolError, match="not a file"):
            await summarize_file_run("subdir", sandbox, mock_llm)

    async def test_sandbox_violation_raises(self, sandbox: Path, mock_llm: AsyncMock) -> None:
        with pytest.raises(SandboxViolationError):
            await summarize_file_run("../secret.txt", sandbox, mock_llm)


# ---------------------------------------------------------------------------
# summarize_text (new tool)
# ---------------------------------------------------------------------------


class TestSummarizeText:
    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        client = AsyncMock()
        client.chat.return_value = "This is a concise summary of the provided text."
        return client

    async def test_returns_llm_summary(self, mock_llm: AsyncMock) -> None:
        result = await summarize_text_run("Some long text to summarize.", llm=mock_llm)
        assert result.summary == "This is a concise summary of the provided text."

    async def test_not_truncated_for_short_text(self, mock_llm: AsyncMock) -> None:
        result = await summarize_text_run("Short text.", llm=mock_llm)
        assert result.truncated is False

    async def test_truncated_for_long_text(self, mock_llm: AsyncMock) -> None:
        long_text = "word " * 2_000  # ~10 000 chars
        result = await summarize_text_run(long_text, max_chars=100, llm=mock_llm)
        assert result.truncated is True

    async def test_llm_called_with_system_and_user_messages(self, mock_llm: AsyncMock) -> None:
        await summarize_text_run("Some text.", llm=mock_llm)
        mock_llm.chat.assert_called_once()
        messages = mock_llm.chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Some text." in messages[1]["content"]

    async def test_empty_llm_response_raises(self, mock_llm: AsyncMock) -> None:
        mock_llm.chat.return_value = "  "
        with pytest.raises(ResponseSynthesisError):
            await summarize_text_run("Some text.", llm=mock_llm)

    async def test_custom_max_chars_respected(self, mock_llm: AsyncMock) -> None:
        text = "a" * 500
        await summarize_text_run(text, max_chars=100, llm=mock_llm)
        messages = mock_llm.chat.call_args[0][0]
        user_content = messages[1]["content"]
        # Content sent to LLM must not exceed max_chars plus the truncation notice
        assert "a" * 101 not in user_content
