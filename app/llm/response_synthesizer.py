"""Pass 2 — LLM response synthesis.

After the MCP tool executes, this module asks the local Ollama LLM to
produce a grounded natural-language response for the user.

Flow:
  1. Build a prompt containing the user's original request and tool output.
  2. Call Ollama (plain text response — no structured output needed here).
  3. Strip whitespace and validate the response is non-empty.
  4. Return the synthesized string.

Grounding rule: the LLM must base its response solely on tool output.
It must not invent facts that are not present in the tool result.
"""

import json
from typing import Any

from app.utils.errors import ResponseSynthesisError
from app.utils.logging import get_logger

log = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant that explains tool results to users in plain language.\n\n"
    "You will receive:\n"
    "  1. The user's original request.\n"
    "  2. The name of the tool that was run.\n"
    "  3. The tool's structured output.\n\n"
    "Your task: write a clear, direct, natural-language response that answers the user's "
    "request using only the information in the tool output.\n\n"
    "Rules:\n"
    "- Do not invent information not present in the tool output.\n"
    "- Keep the response concise and relevant to the user's request.\n"
    "- Format lists or file names using plain text or markdown where helpful.\n"
    "- Do not mention the tool name or internal system details unless relevant."
)


class ResponseSynthesizer:
    """Generates a natural-language response from MCP tool output using a local LLM.

    Args:
        ollama_client: An OllamaClient instance (injected).
    """

    def __init__(self, ollama_client: Any) -> None:
        self._llm = ollama_client

    def build_messages(
        self,
        user_input: str,
        tool_name: str,
        tool_output: Any,
        state_context: dict[str, Any] | None = None,
        context_documents: list[dict[str, Any]] | None = None,
    ) -> list[dict]:
        """Build the Ollama chat messages for the response-synthesis prompt.

        Args:
            user_input:         The original user request.
            tool_name:          Name of the MCP tool that produced the output.
            tool_output:        Structured output returned by the tool.
            state_context:      Optional extra fields from graph state.
            context_documents:  Optional list of fetched external resources
                                [{"url": ..., "content": ...}] from retrieve_context.

        Returns:
            A list of OpenAI-style message dicts ready to send to Ollama.
        """
        tool_output_str = (
            tool_output
            if isinstance(tool_output, str)
            else json.dumps(tool_output, indent=2, default=str)
        )

        user_content = (
            f"User request: {user_input}\n\n"
            f"Tool used: {tool_name}\n\n"
            f"Tool output:\n```json\n{tool_output_str}\n```"
        )

        if context_documents:
            docs_with_content = [d for d in context_documents if d.get("content")]
            if docs_with_content:
                sections: list[str] = []
                for doc in docs_with_content:
                    sections.append(f"Source: {doc['url']}\n\n{doc['content']}")
                user_content += (
                    "\n\n---\nExternal reference material retrieved for this topic:\n\n"
                    + "\n\n---\n".join(sections)
                )

        if state_context:
            user_content += (
                f"\n\nAdditional context:\n"
                f"```json\n{json.dumps(state_context, indent=2, default=str)}\n```"
            )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def synthesize(
        self,
        user_input: str,
        tool_name: str,
        tool_output: Any,
        state_context: dict[str, Any] | None = None,
        model: str | None = None,
        context_documents: list[dict[str, Any]] | None = None,
    ) -> str:
        """Produce a natural-language response grounded in tool output.

        Args:
            user_input:         The original user request.
            tool_name:          Name of the MCP tool that was called.
            tool_output:        Structured output returned by the MCP tool.
            state_context:      Optional additional fields from graph state.
            model:              Optional Ollama model name override.
            context_documents:  Optional list of external resource documents
                                to include as reference material in the prompt.

        Returns:
            A natural-language string suitable for presenting to the user.

        Raises:
            ResponseSynthesisError: If the LLM returns an empty or whitespace-only response.
        """
        messages = self.build_messages(
            user_input, tool_name, tool_output, state_context, context_documents
        )
        log.info(
            "response_synthesizer_start",
            user_input=user_input[:120],
            tool_name=tool_name,
            model=model,
            context_doc_count=len(context_documents) if context_documents else 0,
        )

        raw = await self._llm.chat(messages, model=model)
        response = raw.strip()

        if not response:
            log.error("response_synthesizer_empty", user_input=user_input, tool_name=tool_name)
            raise ResponseSynthesisError(
                f"LLM returned an empty response when synthesizing output from '{tool_name}'"
            )

        log.info("response_synthesizer_complete", length=len(response))
        return response
