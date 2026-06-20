"""Environment-backed settings loaded via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Values are read from environment variables or .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # MCP server
    mcp_server_host: str = "0.0.0.0"
    mcp_server_port: int = 8000

    # SQLite
    sqlite_db_path: str = "data/workflow.db"

    # Redis / message queue
    mq_enabled: bool = False
    redis_url: str = "redis://localhost:6379"
    mq_request_queue: str = "requests"
    mq_response_queue: str = "responses"

    # File system sandbox root for MCP tools
    sandbox_root: str = "./sandbox"

    # Logging
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
