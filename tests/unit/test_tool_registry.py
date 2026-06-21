"""Unit tests for ToolRegistry — thread-safe hot-reload registry."""

import threading

import pytest

from app.llm.tool_registry import ToolRegistry
from app.llm.tool_selector import ToolDefinition


class TestToolRegistryReload:
    def test_reload_populates_definitions(self):
        r = ToolRegistry()
        r.reload()
        assert len(r.definitions) >= 10  # 4 + 4 + 2

    def test_definitions_are_tool_definition_instances(self):
        r = ToolRegistry()
        r.reload()
        for d in r.definitions:
            assert isinstance(d, ToolDefinition)

    def test_all_known_tool_names_present(self):
        r = ToolRegistry()
        r.reload()
        names = {d.name for d in r.definitions}
        expected = {
            "list_files", "read_file", "search_files", "extract_metadata",
            "summarize_file", "summarize_text",
            "get_topic_resources", "fetch_web_resource", "get_cached_resource", "refresh_cache",
        }
        assert expected <= names

    def test_specs_by_name_contains_all_tools(self):
        r = ToolRegistry()
        r.reload()
        specs = r.specs_by_name
        assert "list_files" in specs
        assert "refresh_cache" in specs
        assert "summarize_text" in specs

    def test_specs_by_name_category_is_correct(self):
        r = ToolRegistry()
        r.reload()
        specs = r.specs_by_name
        assert specs["list_files"].category == "file_ops"
        assert specs["fetch_web_resource"].category == "context_retrieval"
        assert specs["summarize_text"].category == "summarization"

    def test_categories_property_structure(self):
        r = ToolRegistry()
        cats = r.reload()
        assert "file_ops" in cats
        assert "context_retrieval" in cats
        assert "summarization" in cats
        assert isinstance(cats["file_ops"], list)

    def test_reload_returns_category_summary(self):
        r = ToolRegistry()
        cats = r.reload()
        total_from_return = sum(len(v) for v in cats.values())
        total_from_property = len(r.definitions)
        assert total_from_return == total_from_property

    def test_reload_is_idempotent(self):
        r = ToolRegistry()
        cats1 = r.reload()
        cats2 = r.reload()
        assert cats1 == cats2

    def test_definitions_returns_snapshot_not_internal_list(self):
        r = ToolRegistry()
        r.reload()
        snapshot = r.definitions
        snapshot.clear()
        # Original must be unaffected
        assert len(r.definitions) > 0

    def test_specs_by_name_returns_snapshot_not_internal_dict(self):
        r = ToolRegistry()
        r.reload()
        snapshot = r.specs_by_name
        snapshot.clear()
        assert len(r.specs_by_name) > 0


class TestToolRegistryBeforeReload:
    def test_definitions_empty_before_reload(self):
        r = ToolRegistry()
        assert r.definitions == []

    def test_specs_by_name_empty_before_reload(self):
        r = ToolRegistry()
        assert r.specs_by_name == {}

    def test_categories_empty_before_reload(self):
        r = ToolRegistry()
        assert r.categories == {}


class TestToolRegistryThreadSafety:
    def test_concurrent_reloads_do_not_corrupt_state(self):
        r = ToolRegistry()
        r.reload()
        errors: list[Exception] = []

        def worker():
            try:
                r.reload()
                _ = r.definitions
                _ = r.specs_by_name
                _ = r.categories
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_reads_while_reload_in_progress(self):
        r = ToolRegistry()
        r.reload()
        results: list[int] = []
        errors: list[Exception] = []

        def reader():
            try:
                for _ in range(20):
                    results.append(len(r.definitions))
            except Exception as exc:
                errors.append(exc)

        def reloader():
            try:
                for _ in range(5):
                    r.reload()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads += [threading.Thread(target=reloader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        # All reads should return non-negative counts
        assert all(n >= 0 for n in results)
