"""Shared ToolSpec dataclass — the contract between tool packages and the loader."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel


@dataclass
class ToolSpec:
    """Describes a single registered MCP tool.

    Attributes:
        name:               Exact tool name (matches executor dispatch key).
        category:           Category folder name (e.g. "file_ops").
        description:        One-line human-readable description for the LLM prompt.
        input_schema_class: Pydantic model class for input validation.
        handler:            Async callable implementing the tool logic.
        dependencies:       Maps handler parameter names to other tool names that
                            the executor must resolve at dispatch time.
                            Example: {"fetcher": "fetch_web_resource"}
    """

    name: str
    category: str
    description: str
    input_schema_class: type[BaseModel]
    handler: Callable[..., Awaitable[Any]]
    dependencies: dict[str, str] = field(default_factory=dict)
