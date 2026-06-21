"""Unit tests for the dynamic tool loader.

Tests use monkeypatching and temporary directories to avoid depending on the
real tools directory — except for the "real tools" integration-style tests at
the bottom which verify all 10 expected tools are discovered correctly.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from app.mcp_server.tool_loader import discover_all_tools, discover_categories, discover_tools


# ---------------------------------------------------------------------------
# discover_categories
# ---------------------------------------------------------------------------


class TestDiscoverCategories:
    def test_returns_only_directories(self, tmp_path: Path, monkeypatch):
        (tmp_path / "file_ops").mkdir()
        (tmp_path / "summarization").mkdir()
        (tmp_path / "_sandbox.py").touch()
        (tmp_path / "README.md").touch()

        import app.mcp_server.tool_loader as loader

        monkeypatch.setattr(loader, "_TOOLS_DIR", tmp_path)

        names = [p.name for p in discover_categories()]
        assert "file_ops" in names
        assert "summarization" in names
        assert "_sandbox.py" not in names
        assert "README.md" not in names

    def test_excludes_underscore_dirs(self, tmp_path: Path, monkeypatch):
        (tmp_path / "file_ops").mkdir()
        (tmp_path / "_private").mkdir()

        import app.mcp_server.tool_loader as loader

        monkeypatch.setattr(loader, "_TOOLS_DIR", tmp_path)

        names = [p.name for p in discover_categories()]
        assert "_private" not in names
        assert "file_ops" in names

    def test_sorted_alphabetically(self, tmp_path: Path, monkeypatch):
        (tmp_path / "zzz_cat").mkdir()
        (tmp_path / "aaa_cat").mkdir()
        (tmp_path / "mmm_cat").mkdir()

        import app.mcp_server.tool_loader as loader

        monkeypatch.setattr(loader, "_TOOLS_DIR", tmp_path)

        names = [p.name for p in discover_categories()]
        assert names == sorted(names)

    def test_empty_dir_returns_empty(self, tmp_path: Path, monkeypatch):
        import app.mcp_server.tool_loader as loader

        monkeypatch.setattr(loader, "_TOOLS_DIR", tmp_path)

        assert discover_categories() == []


# ---------------------------------------------------------------------------
# discover_tools
# ---------------------------------------------------------------------------


class TestDiscoverTools:
    def _make_fake_spec(self, tool_name: str, category: str):
        """Return a ToolSpec-like mock for use with import mocking."""
        from unittest.mock import MagicMock
        from pydantic import BaseModel

        class _In(BaseModel):
            x: int = 0

        async def _run(x: int = 0): ...

        from app.mcp_server.tool_spec import ToolSpec
        return ToolSpec(tool_name, category, "desc", _In, _run)

    def test_discovers_valid_tool(self, tmp_path: Path, monkeypatch):
        """discover_tools calls get_tool() on each tool package and returns its spec."""
        from unittest.mock import MagicMock
        import importlib

        cat_dir = tmp_path / "my_cat"
        tool_dir = cat_dir / "my_tool"
        tool_dir.mkdir(parents=True)

        expected_spec = self._make_fake_spec("my_tool", "my_cat")
        fake_module = MagicMock()
        fake_module.get_tool.return_value = expected_spec

        # Patch importlib.import_module so discover_tools gets our fake module.
        original_import = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "app.mcp_server.tools.my_cat.my_tool":
                return fake_module
            return original_import(name, *args, **kwargs)

        import app.mcp_server.tool_loader as loader

        monkeypatch.setattr(loader, "_TOOLS_DIR", tmp_path)
        monkeypatch.setattr(importlib, "import_module", fake_import)

        specs = discover_tools(cat_dir)
        assert len(specs) == 1
        assert specs[0].name == "my_tool"
        assert specs[0].category == "my_cat"

    def test_skips_underscore_dirs(self, tmp_path: Path):
        cat_dir = tmp_path / "cat"
        (cat_dir / "_shared").mkdir(parents=True)
        (cat_dir / "_shared" / "__init__.py").write_text("", encoding="utf-8")

        specs = discover_tools(cat_dir)
        assert specs == []

    def test_broken_module_is_skipped_not_raised(self, tmp_path: Path, monkeypatch):
        """A tool whose import raises must be logged and skipped, never propagated."""
        import importlib

        cat_dir = tmp_path / "cat"
        bad_dir = cat_dir / "bad_tool"
        bad_dir.mkdir(parents=True)

        original_import = importlib.import_module

        def failing_import(name, *args, **kwargs):
            if name == "app.mcp_server.tools.cat.bad_tool":
                raise ImportError("deliberate test error")
            return original_import(name, *args, **kwargs)

        import app.mcp_server.tool_loader as loader

        monkeypatch.setattr(loader, "_TOOLS_DIR", tmp_path)
        monkeypatch.setattr(importlib, "import_module", failing_import)

        specs = discover_tools(cat_dir)  # must not raise
        assert specs == []

    def test_multiple_tools_in_category(self, tmp_path: Path, monkeypatch):
        """All tool subdirectories in a category are returned as specs."""
        import importlib

        cat_dir = tmp_path / "ops"
        tool_names = ["tool_a", "tool_b", "tool_c"]
        for name in tool_names:
            (cat_dir / name).mkdir(parents=True)

        fake_specs = {n: self._make_fake_spec(n, "ops") for n in tool_names}
        original_import = importlib.import_module

        def multi_import(name, *args, **kwargs):
            for tool_name in tool_names:
                if name == f"app.mcp_server.tools.ops.{tool_name}":
                    from unittest.mock import MagicMock
                    m = MagicMock()
                    m.get_tool.return_value = fake_specs[tool_name]
                    return m
            return original_import(name, *args, **kwargs)

        import app.mcp_server.tool_loader as loader

        monkeypatch.setattr(loader, "_TOOLS_DIR", tmp_path)
        monkeypatch.setattr(importlib, "import_module", multi_import)

        specs = discover_tools(cat_dir)
        assert len(specs) == 3
        names = {s.name for s in specs}
        assert names == {"tool_a", "tool_b", "tool_c"}

    def test_tools_sorted_by_directory_name(self, tmp_path: Path, monkeypatch):
        """Tools within a category are returned sorted alphabetically by folder name."""
        import importlib

        cat_dir = tmp_path / "ops"
        tool_names = ["zzz", "aaa", "mmm"]
        for name in tool_names:
            (cat_dir / name).mkdir(parents=True)

        fake_specs = {n: self._make_fake_spec(n, "ops") for n in tool_names}
        original_import = importlib.import_module

        def sorted_import(name, *args, **kwargs):
            for tool_name in tool_names:
                if name == f"app.mcp_server.tools.ops.{tool_name}":
                    from unittest.mock import MagicMock
                    m = MagicMock()
                    m.get_tool.return_value = fake_specs[tool_name]
                    return m
            return original_import(name, *args, **kwargs)

        import app.mcp_server.tool_loader as loader

        monkeypatch.setattr(loader, "_TOOLS_DIR", tmp_path)
        monkeypatch.setattr(importlib, "import_module", sorted_import)

        specs = discover_tools(cat_dir)
        names = [s.name for s in specs]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# discover_all_tools — uses the real tools directory
# ---------------------------------------------------------------------------


class TestDiscoverAllToolsReal:
    """These tests verify the actual project tool tree is correctly structured."""

    def test_returns_all_three_categories(self):
        result = discover_all_tools()
        assert "file_ops" in result
        assert "context_retrieval" in result
        assert "summarization" in result

    def test_file_ops_has_four_tools(self):
        result = discover_all_tools()
        names = {s.name for s in result["file_ops"]}
        assert names == {"list_files", "read_file", "search_files", "extract_metadata"}

    def test_context_retrieval_has_four_tools(self):
        result = discover_all_tools()
        names = {s.name for s in result["context_retrieval"]}
        assert names == {
            "get_topic_resources",
            "fetch_web_resource",
            "get_cached_resource",
            "refresh_cache",
        }

    def test_summarization_has_two_tools(self):
        result = discover_all_tools()
        names = {s.name for s in result["summarization"]}
        assert names == {"summarize_file", "summarize_text"}

    def test_each_spec_category_matches_folder(self):
        result = discover_all_tools()
        for cat_name, specs in result.items():
            for spec in specs:
                assert spec.category == cat_name, (
                    f"Tool '{spec.name}' has category '{spec.category}' "
                    f"but was discovered in folder '{cat_name}'"
                )

    def test_refresh_cache_declares_fetcher_dependency(self):
        result = discover_all_tools()
        rc = next(s for s in result["context_retrieval"] if s.name == "refresh_cache")
        assert rc.dependencies.get("fetcher") == "fetch_web_resource"

    def test_total_tool_count(self):
        result = discover_all_tools()
        total = sum(len(v) for v in result.values())
        assert total == 10  # 4 file_ops + 2 summarization + 4 context_retrieval

    def test_all_specs_have_callable_handlers(self):
        result = discover_all_tools()
        for specs in result.values():
            for spec in specs:
                assert callable(spec.handler), f"Tool '{spec.name}' handler is not callable"

    def test_all_specs_have_pydantic_input_schema(self):
        from pydantic import BaseModel

        result = discover_all_tools()
        for specs in result.values():
            for spec in specs:
                assert issubclass(spec.input_schema_class, BaseModel), (
                    f"Tool '{spec.name}' input_schema_class is not a Pydantic BaseModel"
                )

    def test_reload_is_idempotent(self):
        result1 = discover_all_tools()
        result2 = discover_all_tools()
        cats1 = {cat: {s.name for s in specs} for cat, specs in result1.items()}
        cats2 = {cat: {s.name for s in specs} for cat, specs in result2.items()}
        assert cats1 == cats2
