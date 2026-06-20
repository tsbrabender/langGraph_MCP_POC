"""MQ producer — publishes user requests and polls for responses via Redis.

Message flow:
  publish_request()  →  LPUSH  →  request queue  →  consumer
  consumer           →  LPUSH  →  response:<id>
  get_response()     →  BRPOP  →  response:<id>

MQ is optional — this class is only instantiated when MQ_ENABLED=true.
The producer can be used as an async context manager:

    async with MQProducer() as producer:
        msg = await producer.publish_request("list the files")
        response = await producer.get_response(msg.request_id)
"""

from typing import Any

import redis.asyncio as aioredis

from app.services.mq.schemas import RequestMessage, ResponseMessage
from app.utils.config import get_settings
from app.utils.errors import MessageQueueError
from app.utils.logging import get_logger

log = get_logger(__name__)


class MQProducer:
    """Publishes JSON request messages to Redis and retrieves responses.

    Args:
        redis_url:       Redis connection URL (falls back to settings).
        request_queue:   Name of the queue consumers read from.
        _redis_client:   Pre-built Redis client — used in tests to avoid real connections.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        request_queue: str | None = None,
        _redis_client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self._redis_url = redis_url or settings.redis_url
        self._request_queue = request_queue or settings.mq_request_queue
        self._redis: aioredis.Redis | None = _redis_client

    async def connect(self) -> None:
        """Open the Redis connection (no-op if a client was pre-injected)."""
        if self._redis is None:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        log.info("mq_producer_connected", url=self._redis_url, queue=self._request_queue)

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            log.info("mq_producer_disconnected")

    def _client(self) -> aioredis.Redis:
        if self._redis is None:
            raise MessageQueueError("MQProducer is not connected. Call connect() first.")
        return self._redis

    async def publish_request(
        self,
        user_input: str,
        metadata: dict[str, Any] | None = None,
    ) -> RequestMessage:
        """Serialize and push a request onto the request queue.

        Args:
            user_input: The user's raw input string.
            metadata:   Optional dict of additional context.

        Returns:
            The RequestMessage that was enqueued (contains the request_id).

        Raises:
            MessageQueueError: If the publish fails.
        """
        message = RequestMessage(user_input=user_input, metadata=metadata or {})
        payload = message.model_dump_json()
        try:
            await self._client().lpush(self._request_queue, payload)
            log.info("mq_request_published", request_id=message.request_id)
            return message
        except Exception as exc:
            raise MessageQueueError(f"Failed to publish request: {exc}") from exc

    async def get_response(
        self,
        request_id: str,
        timeout: float = 30.0,
    ) -> ResponseMessage | None:
        """Block until a response is available for request_id or timeout elapses.

        The consumer publishes responses to the key ``response:<request_id>``.
        This method uses BRPOP so it does not busy-poll Redis.

        Args:
            request_id: The request_id returned by publish_request().
            timeout:    Maximum seconds to wait. Returns None on timeout.

        Returns:
            A ResponseMessage, or None if the timeout elapsed.

        Raises:
            MessageQueueError: If the Redis call fails.
        """
        response_key = f"response:{request_id}"
        try:
            result = await self._client().brpop(response_key, timeout=int(timeout))
            if result is None:
                log.warning("mq_response_timeout", request_id=request_id)
                return None
            _, raw = result
            response = ResponseMessage.model_validate_json(raw)
            log.info("mq_response_received", request_id=request_id)
            return response
        except Exception as exc:
            raise MessageQueueError(f"Failed to get response for {request_id!r}: {exc}") from exc

    async def __aenter__(self) -> "MQProducer":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()
