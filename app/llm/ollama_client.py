"""Wrapper for local Ollama LLM calls.

Provides a thin async interface over the ollama-python client.
Step 2 will use this client in tool_selector.py and response_synthesizer.py.
"""

from typing import Any
import ollama

from app.utils.config import get_settings
from app.utils.errors import LLMError
from app.utils.logging import get_logger

log = get_logger(__name__)


class OllamaClient:
    """Async wrapper around the Ollama Python client."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        settings = get_settings()
        self._base_url = base_url or settings.ollama_base_url
        self._model = model or settings.ollama_model
        self._client = ollama.AsyncClient(host=self._base_url)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        format: Any = None,
    ) -> str:
        """Send a chat request to Ollama and return the assistant message content.

        Args:
            messages: OpenAI-style message list [{"role": ..., "content": ...}].
            model: Override the default model for this call.
            format: Optional response format (e.g. a JSON schema for structured output).

        Returns:
            The assistant's response as a plain string.

        Raises:
            LLMError: If the Ollama call fails.
        """
        target_model = model or self._model
        log.info("ollama_chat", model=target_model, message_count=len(messages))
        try:
            kwargs: dict[str, Any] = {"model": target_model, "messages": messages}
            if format is not None:
                kwargs["format"] = format
            response = await self._client.chat(**kwargs)
            content: str = response["message"]["content"]
            log.info("ollama_chat_complete", model=target_model, response_length=len(content))
            return content
        except Exception as exc:
            log.error("ollama_chat_failed", model=target_model, error=str(exc))
            raise LLMError(f"Ollama call failed: {exc}") from exc

    async def health_check(self) -> bool:
        """Return True if the Ollama service is reachable and the model is available."""
        try:
            models = await self._client.list()
            available = [m["model"] for m in models.get("models", [])]
            return self._model in available
        except Exception:
            return False
