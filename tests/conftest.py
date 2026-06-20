"""Shared pytest fixtures available to all test modules."""

import pytest
from unittest.mock import AsyncMock

from app.utils.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Return a test Settings instance with safe defaults."""
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.2",
        sqlite_db_path=":memory:",
        mq_enabled=False,
        sandbox_root="./tests/fixtures/sandbox",
        log_level="DEBUG",
    )


@pytest.fixture
def base_state() -> dict:
    """Return a minimal GraphState dict for use in node unit tests."""
    return {"user_input": "list the files in the sandbox"}


@pytest.fixture
def mock_ollama_client() -> AsyncMock:
    """Return a mock OllamaClient that returns an empty string by default."""
    client = AsyncMock()
    client.chat.return_value = ""
    client.health_check.return_value = True
    return client
