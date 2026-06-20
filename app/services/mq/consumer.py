"""MQ consumer — processes requests from Redis and publishes responses.

Message flow:
  BRPOP request queue  →  validate  →  dedup check  →  run LangGraph
      →  build ResponseMessage  →  LPUSH response:<id>  →  expire key

Idempotency:
  Each request_id is written to a Redis key (mq:processed:<id>) with NX
  semantics before processing. If the key already exists the message is
  a duplicate and is silently discarded.

MQ is optional — MQConsumer is only instantiated when MQ_ENABLED=true.
"""

from typing import Any

import redis.asyncio as aioredis

from app.services.mq.schemas import RequestMessage, ResponseMessage
from app.utils.config import get_settings
from app.utils.errors import MessageQueueError
from app.utils.logging import get_logger

log = get_logger(__name__)

_DEDUP_TTL = 3600        # Seconds to remember processed request IDs.
_RESPONSE_TTL = 3600     # Seconds before per-request response keys expire.


class MQConsumer:
    """Consumes request messages, runs the LangGraph workflow, and publishes responses.

    Args:
        graph:          Compiled LangGraph runnable (from build_llm_graph or build_hybrid_graph).
        redis_url:      Redis connection URL (falls back to settings).
        request_queue:  Name of the queue to consume from.
        response_ttl:   Seconds before per-request response keys are auto-deleted.
        _redis_client:  Pre-built Redis client — used in tests to avoid real connections.
    """

    def __init__(
        self,
        graph: Any,
        redis_url: str | None = None,
        request_queue: str | None = None,
        response_ttl: int = _RESPONSE_TTL,
        _redis_client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self._graph = graph
        self._redis_url = redis_url or settings.redis_url
        self._request_queue = request_queue or settings.mq_request_queue
        self._response_ttl = response_ttl
        self._redis: aioredis.Redis | None = _redis_client
        self._running = False

    async def connect(self) -> None:
        """Open the Redis connection (no-op if a client was pre-injected)."""
        if self._redis is None:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        log.info("mq_consumer_connected", queue=self._request_queue)

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            log.info("mq_consumer_disconnected")

    def _client(self) -> aioredis.Redis:
        if self._redis is None:
            raise MessageQueueError("MQConsumer is not connected. Call connect() first.")
        return self._redis

    async def start(self) -> None:
        """Connect and enter the blocking BRPOP consume loop.

        Runs until stop() is called. Handles individual message errors gracefully
        without crashing the loop.
        """
        await self.connect()
        self._running = True
        log.info("mq_consumer_started", queue=self._request_queue)
        try:
            while self._running:
                result = await self._client().brpop(self._request_queue, timeout=1)
                if result is None:
                    continue  # Timeout — check self._running and loop
                _, raw = result
                try:
                    await self.process_message(raw)
                except Exception as exc:
                    log.error("mq_consumer_unhandled_error", error=str(exc))
        finally:
            await self.disconnect()

    async def stop(self) -> None:
        """Signal the consume loop to exit after the current iteration."""
        self._running = False
        log.info("mq_consumer_stopped")

    async def _is_new_message(self, request_id: str) -> bool:
        """Return True only if this request_id has not been processed before.

        Uses Redis SET NX to atomically claim the ID. The key expires after
        _DEDUP_TTL seconds to avoid unbounded key growth.
        """
        key = f"mq:processed:{request_id}"
        result = await self._client().set(key, "1", ex=_DEDUP_TTL, nx=True)
        return result is not None  # None → key existed → duplicate

    async def process_message(self, raw: str) -> None:
        """Parse one raw JSON message, deduplicate, run the graph, publish response.

        This method is intentionally public so unit tests can drive it directly
        without the blocking BRPOP loop.

        Malformed JSON and duplicate messages are silently discarded.
        Graph errors are caught and surfaced as error responses.

        Args:
            raw: Raw JSON string from Redis.
        """
        # --- Parse ---
        try:
            request = RequestMessage.model_validate_json(raw)
        except Exception as exc:
            log.error("mq_consumer_parse_error", raw=raw[:200], error=str(exc))
            return  # Discard — cannot even extract a request_id to ack

        log.info("mq_consumer_received", request_id=request.request_id)

        # --- Idempotency ---
        if not await self._is_new_message(request.request_id):
            log.warning("mq_consumer_duplicate_skipped", request_id=request.request_id)
            return

        # --- Run graph ---
        final_state: dict[str, Any]
        try:
            initial_state: dict[str, Any] = {
                "user_input": request.user_input,
                "metadata": {**request.metadata, "request_id": request.request_id},
            }
            final_state = await self._graph.ainvoke(initial_state)
        except Exception as exc:
            log.error("mq_consumer_graph_error", request_id=request.request_id, error=str(exc))
            final_state = {"error": str(exc), "final_response": None}

        # --- Build response ---
        final_response = final_state.get("final_response") or (
            f"An error occurred: {final_state.get('error', 'unknown error')}"
        )
        response = ResponseMessage(
            request_id=request.request_id,
            final_response=final_response,
            selected_tool=final_state.get("selected_tool"),
            tool_output=final_state.get("tool_output"),
            error=final_state.get("error"),
        )

        # --- Publish response ---
        response_key = f"response:{request.request_id}"
        client = self._client()
        await client.lpush(response_key, response.model_dump_json())
        await client.expire(response_key, self._response_ttl)

        log.info("mq_consumer_response_published", request_id=request.request_id)

    async def __aenter__(self) -> "MQConsumer":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()
