"""Tool registry — builds ToolDefinition instances from all registered MCP tools.

This module is the explicit connector between the LLM layer (app/llm/) and the
MCP tool layer (app/mcp_server/tools/). It is the only file that imports from both.
Graph nodes and LLM classes must not import from this module's dependencies directly.
"""

from app.llm.tool_selector import ToolDefinition
from app.mcp_server.tools.extract_metadata import ExtractMetadataInput
from app.mcp_server.tools.list_files import ListFilesInput
from app.mcp_server.tools.read_file import ReadFileInput
from app.mcp_server.tools.search_files import SearchFilesInput
from app.mcp_server.tools.summarize_file import SummarizeFileInput


def build_tool_definitions() -> list[ToolDefinition]:
    """Return the full list of ToolDefinition objects for all registered MCP tools.

    Add a new entry here whenever a new MCP tool is registered in server.py.
    """
    return [
        ToolDefinition(
            name="list_files",
            description=(
                "List files and subdirectories at the given path within the sandbox. "
                "Use when the user wants to see what files exist in a directory."
            ),
            input_schema_class=ListFilesInput,
        ),
        ToolDefinition(
            name="read_file",
            description=(
                "Read and return the full text content of a specific file within the sandbox. "
                "Use when the user wants to see or inspect the contents of a named file."
            ),
            input_schema_class=ReadFileInput,
        ),
        ToolDefinition(
            name="search_files",
            description=(
                "Search for files matching a glob pattern within the sandbox. "
                "Supports recursive patterns like '**/*.py'. "
                "Use when the user wants to find files by name or extension."
            ),
            input_schema_class=SearchFilesInput,
        ),
        ToolDefinition(
            name="summarize_file",
            description=(
                "Summarize the content of a file using the local LLM. "
                "Use when the user wants a brief overview of a file without reading it in full."
            ),
            input_schema_class=SummarizeFileInput,
        ),
        ToolDefinition(
            name="extract_metadata",
            description=(
                "Return file system metadata for a path: name, extension, size, "
                "timestamps, and whether it is a file or directory. "
                "Use when the user asks about file details, not file content."
            ),
            input_schema_class=ExtractMetadataInput,
        ),
    ]
