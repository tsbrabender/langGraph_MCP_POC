"""Pass 1 — LLM-driven MCP tool selection.

Given user input and a registered set of tool definitions, asks the local
Ollama LLM to choose the best tool and produce validated arguments.

Flow:
  1. Build a system prompt containing all tool descriptions + JSON schemas.
  2. Call Ollama with a structured-output schema that forces valid JSON.
  3. Parse the JSON response.
  4. Verify the selected tool name exists in the registry.
  5. Validate the arguments against the tool's Pydantic input schema.
  6. Return a ToolCall with the validated name and arguments.
"""

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.utils.errors import ToolSelectionError
from app.utils.logging import get_logger

log = get_logger(__name__)

# Ollama structured-output schema: forces the model to return exactly this shape.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string", "description": "Exact name of the MCP tool to call."},
        "arguments": {"type": "object", "description": "Arguments for the selected tool."},
        "reasoning": {"type": "string", "description": "One sentence explaining the choice."},
    },
    "required": ["tool_name", "arguments", "reasoning"],
    "additionalProperties": False,
}

_EXAMPLES = [
    {
        "tool_name": "list_files",
        "arguments": {"directory": "."},
        "reasoning": "User wants to see what files are present in the sandbox root.",
    },
    {
        "tool_name": "read_file",
        "arguments": {"path": "notes.md"},
        "reasoning": "User asked to read a specific named file.",
    },
    {
        "tool_name": "search_files",
        "arguments": {"pattern": "**/*.py", "directory": "."},
        "reasoning": "User wants to find all Python source files recursively.",
    },
    {
        "tool_name": "extract_metadata",
        "arguments": {"path": "data.json"},
        "reasoning": "User asked for file details such as size and modification date.",
    },
    {
        "tool_name": "summarize_file",
        "arguments": {"path": "report.txt"},
        "reasoning": "User wants a brief overview of the file's content.",
    },
]


@dataclass
class ToolDefinition:
    """Describes a single MCP tool for use in the LLM selection prompt.

    Attributes:
        name: Exact tool name as registered in the FastMCP server.
        description: Human-readable one-line description of what the tool does.
        input_schema_class: Pydantic model class for the tool's input arguments.
    """

    name: str
    description: str
    input_schema_class: type[BaseModel]

    def json_schema(self) -> dict:
        """Return the JSON schema for this tool's input arguments."""
        return self.input_schema_class.model_json_schema()


class ToolCall(BaseModel):
    """Validated output of the tool-selection pass.

    Attributes:
        tool_name: Exact name of the selected MCP tool.
        arguments: Tool arguments validated against the tool's Pydantic schema.
        reasoning: LLM's one-sentence explanation of the selection.
    """

    tool_name: str
    arguments: dict[str, Any]
    reasoning: str


class ToolSelector:
    """Selects an MCP tool and generates validated arguments using a local LLM.

    Args:
        ollama_client: An OllamaClient instance (injected).
        tool_definitions: List of ToolDefinition objects describing available tools.
    """

    def __init__(self, ollama_client: Any, tool_definitions: list[ToolDefinition]) -> None:
        self._llm = ollama_client
        self._tools: dict[str, ToolDefinition] = {t.name: t for t in tool_definitions}

    def build_messages(self, user_input: str, context: dict[str, Any] | None = None) -> list[dict]:
        """Build the Ollama chat messages for the tool-selection prompt.

        Args:
            user_input: Raw user request.
            context: Optional extra fields from graph state (e.g. conversation_history).

        Returns:
            A list of OpenAI-style message dicts ready to send to Ollama.
        """
        tool_blocks: list[str] = []
        for tool in self._tools.values():
            schema_str = json.dumps(tool.json_schema(), indent=2)
            tool_blocks.append(
                f"### {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Input schema:\n```json\n{schema_str}\n```"
            )

        tools_section = "\n\n".join(tool_blocks)
        examples_str = json.dumps(_EXAMPLES, indent=2)

        system_prompt = (
            "You are a tool-selection assistant for a local file-system agent.\n"
            "Given a user request, select the most appropriate tool from the list below "
            "and produce its arguments.\n\n"
            f"## Available tools\n\n{tools_section}\n\n"
            f"## Example responses\n```json\n{examples_str}\n```\n\n"
            "Rules:\n"
            "- Respond only with a single JSON object matching the required schema.\n"
            "- The tool_name must exactly match one of the available tool names.\n"
            "- Arguments must conform to the tool's input schema.\n"
            "- Omit optional arguments if not needed; use defaults where appropriate."
        )

        user_content = f"User request: {user_input}"
        if context:
            user_content += f"\n\nContext:\n{json.dumps(context, indent=2, default=str)}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def select(self, user_input: str, context: dict[str, Any] | None = None) -> ToolCall:
        """Ask the LLM to select a tool and return a validated ToolCall.

        Args:
            user_input: Raw user request string.
            context: Optional additional context from graph state.

        Returns:
            A ToolCall with a validated tool name and arguments.

        Raises:
            ToolSelectionError: If the LLM returns invalid JSON, names an unknown
                tool, or provides arguments that fail Pydantic validation.
        """
        messages = self.build_messages(user_input, context)
        log.info("tool_selector_start", user_input=user_input[:120])

        raw = await self._llm.chat(messages, format=_RESPONSE_SCHEMA)

        # --- Parse JSON ---
        try:
            data: dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.error("tool_selector_json_error", snippet=raw[:200])
            raise ToolSelectionError(f"LLM returned non-JSON: {exc}") from exc

        # --- Validate tool name ---
        tool_name: str = data.get("tool_name", "")
        if tool_name not in self._tools:
            raise ToolSelectionError(
                f"LLM selected unknown tool '{tool_name}'. "
                f"Available: {sorted(self._tools.keys())}"
            )

        # --- Validate arguments against Pydantic schema ---
        raw_args: dict = data.get("arguments", {})
        tool_def = self._tools[tool_name]
        try:
            validated_model = tool_def.input_schema_class(**raw_args)
        except ValidationError as exc:
            raise ToolSelectionError(
                f"Arguments for '{tool_name}' failed validation: {exc}"
            ) from exc

        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=validated_model.model_dump(),
            reasoning=data.get("reasoning", ""),
        )
        log.info("tool_selector_complete", tool_name=tool_name, arguments=tool_call.arguments)
        return tool_call
