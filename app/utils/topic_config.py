"""Topic-to-resource configuration loader.

Reads topic_map.yaml and exposes a validated TopicMap. Each topic defines
the keyword aliases used to detect user intent and the URLs to fetch, plus
an optional per-topic TTL override.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator

from app.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_TOPIC_MAP_PATH = Path(__file__).parent.parent / "mcp_server" / "resources" / "topic_map.yaml"


class TopicEntry(BaseModel):
    """Configuration for a single topic.

    Attributes:
        keywords:    List of strings matched against user input (case-insensitive).
        urls:        Ordered list of URLs to fetch when this topic is detected.
        ttl_seconds: How long fetched content is valid in the cache (default 6 h).
    """

    keywords: list[str]
    urls: list[str]
    ttl_seconds: int = 21600

    @field_validator("keywords")
    @classmethod
    def keywords_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("keywords must contain at least one entry")
        return [kw.lower() for kw in v]

    @field_validator("urls")
    @classmethod
    def urls_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("urls must contain at least one entry")
        return v

    @field_validator("ttl_seconds")
    @classmethod
    def ttl_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("ttl_seconds must be positive")
        return v


class TopicMap(BaseModel):
    """Validated wrapper around the full topic map.

    Attributes:
        topics: Dict mapping topic name (lowercase) to its TopicEntry.
    """

    topics: dict[str, TopicEntry]

    def get_entry(self, topic: str) -> TopicEntry | None:
        """Return the TopicEntry for a topic name (case-insensitive), or None."""
        return self.topics.get(topic.lower())

    def detect_topic(self, user_input: str) -> str | None:
        """Scan user_input for any topic keyword and return the matched topic name.

        Returns the first match found in iteration order (dict insertion order).
        Returns None if no keyword matches.
        """
        lowered = user_input.lower()
        for topic_name, entry in self.topics.items():
            for kw in entry.keywords:
                if kw in lowered:
                    return topic_name
        return None


def _load_raw(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"topic_map.yaml not found at {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "topics" not in data:
        raise ValueError(f"topic_map.yaml must be a mapping with a 'topics' key: {path}")
    return data


@lru_cache(maxsize=1)
def load_topic_map(path: Path | None = None) -> TopicMap:
    """Load and validate the topic map YAML, cached for the process lifetime.

    Args:
        path: Override path to the YAML file. Defaults to the bundled topic_map.yaml.

    Returns:
        A validated TopicMap instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValidationError:   If the YAML structure is invalid.
    """
    resolved = path or _DEFAULT_TOPIC_MAP_PATH
    raw = _load_raw(resolved)
    topic_map = TopicMap(**raw)
    log.info("topic_map_loaded", path=str(resolved), topic_count=len(topic_map.topics))
    return topic_map
