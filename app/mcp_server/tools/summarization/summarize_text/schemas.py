"""Pydantic schemas for the summarize_text tool."""

from pydantic import BaseModel, Field


class SummarizeTextInput(BaseModel):
    text: str = Field(description="The text content to summarize")
    max_chars: int = Field(
        default=4_000,
        description="Maximum characters to send to the LLM (excess is truncated)",
        ge=100,
        le=16_000,
    )


class SummarizeTextOutput(BaseModel):
    summary: str
    truncated: bool
