"""Shared exception types for the LangGraph-MCP system."""


class MCPToolError(Exception):
    """Raised when an MCP tool fails to execute."""


class SandboxViolationError(MCPToolError):
    """Raised when a tool attempts to access a path outside the sandbox root."""


class LLMError(Exception):
    """Raised when a local LLM call fails."""


class ToolSelectionError(LLMError):
    """Raised when the LLM fails to produce a valid tool selection."""


class ResponseSynthesisError(LLMError):
    """Raised when the LLM fails to produce a valid natural-language response."""


class DatabaseError(Exception):
    """Raised when a SQLite operation fails."""


class MessageQueueError(Exception):
    """Raised when a message queue publish or consume operation fails."""


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
