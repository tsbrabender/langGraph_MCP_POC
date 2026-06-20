"""Unit tests for MQ schemas, producer, and consumer.

All tests use mocked Redis clients — no live Redis instance required.
The consumer's process_message() is tested directly to avoid the blocking
BRPOP loop, which requires a real event loop and live queue.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.mq.schemas import RequestMessage, ResponseMessage
from app.services.mq.producer import MQProducer
from app.services.mq.consumer import MQConsumer
from app.utils.errors import MessageQueueError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request_json(user_input: str = "list files", request_id: str = "test-id-123") -> str:
    msg = RequestMessage(request_id=request_id, user_input=user_input)
    return msg.model_dump_json()


def _graph_state(
    final_response: str = "Here are the files.",
    tool: str = "list_files",
) -> dict:
    return {
        "user_input": "list files",
        "selected_tool": tool,
        "tool_output": {"count": 3, "entries": [], "directory": "."},
        "final_response": final_response,
        "conversation_history": [],
        "metadata": {},
        "error": None,
    }


# ---------------------------------------------------------------------------
# RequestMessage schema
# ---------------------------------------------------------------------------


class TestRequestMessage:
    def test_auto_generates_request_id(self) -> None:
        msg = RequestMessage(user_input="hello")
        assert msg.request_id
        assert len(msg.request_id) == 36  # UUID4 format

    def test_request_ids_are_unique(self) -> None:
        a = RequestMessage(user_input="hello")
        b = RequestMessage(user_input="hello")
        assert a.request_id != b.request_id

    def test_created_at_is_iso8601(self) -> None:
        msg = RequestMessage(user_input="hello")
        assert "T" in msg.created_at
        assert "+" in msg.created_at or "Z" in msg.created_at or msg.created_at.endswith("+00:00")

    def test_metadata_defaults_to_empty_dict(self) -> None:
        msg = RequestMessage(user_input="hello")
        assert msg.metadata == {}

    def test_serializes_and_deserializes(self) -> None:
        original = RequestMessage(user_input="find *.py", metadata={"source": "test"})
        raw = original.model_dump_json()
        restored = RequestMessage.model_validate_json(raw)
        assert restored.request_id == original.request_id
        assert restored.user_input == original.user_input
        assert restored.metadata == {"source": "test"}

    def test_explicit_request_id_preserved(self) -> None:
        msg = RequestMessage(request_id="my-id", user_input="hello")
        assert msg.request_id == "my-id"


# ---------------------------------------------------------------------------
# ResponseMessage schema
# ---------------------------------------------------------------------------


class TestResponseMessage:
    def test_required_fields(self) -> None:
        msg = ResponseMessage(request_id="abc", final_response="Done.")
        assert msg.request_id == "abc"
        assert msg.final_response == "Done."

    def test_optional_fields_default_to_none(self) -> None:
        msg = ResponseMessage(request_id="abc", final_response="Done.")
        assert msg.selected_tool is None
        assert msg.tool_output is None
        assert msg.error is None

    def test_completed_at_auto_set(self) -> None:
        msg = ResponseMessage(request_id="x", final_response="ok")
        assert "T" in msg.completed_at

    def test_serializes_with_tool_output(self) -> None:
        msg = ResponseMessage(
            request_id="x",
            final_response="ok",
            selected_tool="list_files",
            tool_output={"count": 2},
        )
        raw = msg.model_dump_json()
        restored = ResponseMessage.model_validate_json(raw)
        assert restored.tool_output == {"count": 2}
        assert restored.selected_tool == "list_files"

    def test_error_field_propagated(self) -> None:
        msg = ResponseMessage(request_id="x", final_response="err", error="tool failed")
        raw = msg.model_dump_json()
        restored = ResponseMessage.model_validate_json(raw)
        assert restored.error == "tool failed"


# ---------------------------------------------------------------------------
# MQProducer
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis() -> AsyncMock:
    r = AsyncMock()
    r.lpush.return_value = 1
    r.brpop.return_value = None
    r.aclose.return_value = None
    return r


@pytest.fixture
def producer(mock_redis: AsyncMock) -> MQProducer:
    return MQProducer(request_queue="requests", _redis_client=mock_redis)


class TestMQProducer:
    async def test_publish_request_returns_request_message(
        self, producer: MQProducer
    ) -> None:
        msg = await producer.publish_request("list the files")
        assert isinstance(msg, RequestMessage)
        assert msg.user_input == "list the files"

    async def test_publish_request_calls_lpush(
        self, producer: MQProducer, mock_redis: AsyncMock
    ) -> None:
        msg = await producer.publish_request("list files")
        mock_redis.lpush.assert_called_once()
        queue_name, payload = mock_redis.lpush.call_args[0]
        assert queue_name == "requests"
        parsed = json.loads(payload)
        assert parsed["user_input"] == "list files"
        assert parsed["request_id"] == msg.request_id

    async def test_publish_request_includes_metadata(
        self, producer: MQProducer, mock_redis: AsyncMock
    ) -> None:
        await producer.publish_request("x", metadata={"source": "ui"})
        _, payload = mock_redis.lpush.call_args[0]
        parsed = json.loads(payload)
        assert parsed["metadata"]["source"] == "ui"

    async def test_publish_request_raises_on_redis_error(
        self, mock_redis: AsyncMock
    ) -> None:
        mock_redis.lpush.side_effect = Exception("connection refused")
        producer = MQProducer(_redis_client=mock_redis)
        with pytest.raises(MessageQueueError, match="Failed to publish"):
            await producer.publish_request("hello")

    async def test_get_response_returns_response_message(
        self, mock_redis: AsyncMock
    ) -> None:
        response = ResponseMessage(
            request_id="test-id",
            final_response="Found 3 files.",
            selected_tool="list_files",
        )
        mock_redis.brpop.return_value = ("response:test-id", response.model_dump_json())
        producer = MQProducer(_redis_client=mock_redis)
        result = await producer.get_response("test-id", timeout=1)
        assert result is not None
        assert result.final_response == "Found 3 files."
        assert result.request_id == "test-id"

    async def test_get_response_returns_none_on_timeout(
        self, producer: MQProducer, mock_redis: AsyncMock
    ) -> None:
        mock_redis.brpop.return_value = None  # Timeout
        result = await producer.get_response("missing-id", timeout=1)
        assert result is None

    async def test_get_response_polls_correct_key(
        self, mock_redis: AsyncMock
    ) -> None:
        mock_redis.brpop.return_value = None
        producer = MQProducer(_redis_client=mock_redis)
        await producer.get_response("abc-123", timeout=1)
        mock_redis.brpop.assert_called_once_with("response:abc-123", timeout=1)

    async def test_get_response_raises_on_redis_error(
        self, mock_redis: AsyncMock
    ) -> None:
        mock_redis.brpop.side_effect = Exception("connection lost")
        producer = MQProducer(_redis_client=mock_redis)
        with pytest.raises(MessageQueueError, match="Failed to get response"):
            await producer.get_response("x", timeout=1)

    async def test_context_manager_connects_and_disconnects(
        self, mock_redis: AsyncMock
    ) -> None:
        producer = MQProducer(_redis_client=mock_redis)
        async with producer as p:
            assert p._redis is mock_redis
        mock_redis.aclose.assert_called_once()

    async def test_not_connected_raises(self) -> None:
        producer = MQProducer()  # No injected client, not connected
        with pytest.raises(MessageQueueError, match="not connected"):
            await producer.publish_request("hello")


# ---------------------------------------------------------------------------
# MQConsumer — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph() -> AsyncMock:
    graph = AsyncMock()
    graph.ainvoke.return_value = _graph_state()
    return graph


@pytest.fixture
def consumer(mock_redis: AsyncMock, mock_graph: AsyncMock) -> MQConsumer:
    return MQConsumer(
        graph=mock_graph,
        request_queue="requests",
        _redis_client=mock_redis,
    )


# ---------------------------------------------------------------------------
# MQConsumer — process_message happy path
# ---------------------------------------------------------------------------


class TestMQConsumerProcessMessage:
    async def test_calls_graph_ainvoke(
        self, consumer: MQConsumer, mock_graph: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True  # Not a duplicate
        raw = make_request_json(user_input="list files")
        await consumer.process_message(raw)
        mock_graph.ainvoke.assert_called_once()
        state_arg = mock_graph.ainvoke.call_args[0][0]
        assert state_arg["user_input"] == "list files"

    async def test_publishes_response_to_correct_key(
        self, consumer: MQConsumer, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True
        raw = make_request_json(request_id="req-abc")
        await consumer.process_message(raw)
        push_call = mock_redis.lpush.call_args
        assert push_call[0][0] == "response:req-abc"

    async def test_response_json_valid(
        self, consumer: MQConsumer, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True
        await consumer.process_message(make_request_json(request_id="req-xyz"))
        _, payload = mock_redis.lpush.call_args[0]
        response = ResponseMessage.model_validate_json(payload)
        assert response.request_id == "req-xyz"
        assert response.final_response == "Here are the files."
        assert response.selected_tool == "list_files"

    async def test_sets_response_key_expiry(
        self, consumer: MQConsumer, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True
        await consumer.process_message(make_request_json(request_id="req-exp"))
        mock_redis.expire.assert_called_once_with("response:req-exp", consumer._response_ttl)

    async def test_request_id_propagated_to_initial_state(
        self, consumer: MQConsumer, mock_graph: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True
        await consumer.process_message(make_request_json(request_id="req-meta"))
        state = mock_graph.ainvoke.call_args[0][0]
        assert state["metadata"]["request_id"] == "req-meta"


# ---------------------------------------------------------------------------
# MQConsumer — idempotency
# ---------------------------------------------------------------------------


class TestMQConsumerIdempotency:
    async def test_duplicate_message_skips_graph(
        self, consumer: MQConsumer, mock_graph: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = None  # NX failed → duplicate
        await consumer.process_message(make_request_json())
        mock_graph.ainvoke.assert_not_called()

    async def test_duplicate_message_skips_publish(
        self, consumer: MQConsumer, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = None
        await consumer.process_message(make_request_json())
        mock_redis.lpush.assert_not_called()

    async def test_dedup_key_uses_correct_prefix(
        self, consumer: MQConsumer, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True
        await consumer.process_message(make_request_json(request_id="dedup-id"))
        set_call = mock_redis.set.call_args
        key_arg = set_call[0][0]
        assert key_arg == "mq:processed:dedup-id"

    async def test_dedup_uses_nx_flag(
        self, consumer: MQConsumer, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True
        await consumer.process_message(make_request_json())
        set_call = mock_redis.set.call_args
        assert set_call[1].get("nx") is True

    async def test_dedup_sets_expiry(
        self, consumer: MQConsumer, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True
        await consumer.process_message(make_request_json())
        set_call = mock_redis.set.call_args
        assert set_call[1].get("ex") > 0


# ---------------------------------------------------------------------------
# MQConsumer — error handling
# ---------------------------------------------------------------------------


class TestMQConsumerErrorHandling:
    async def test_malformed_json_does_not_raise(
        self, consumer: MQConsumer, mock_graph: AsyncMock
    ) -> None:
        await consumer.process_message("not-valid-json{{{")
        mock_graph.ainvoke.assert_not_called()

    async def test_invalid_schema_does_not_raise(
        self, consumer: MQConsumer, mock_graph: AsyncMock
    ) -> None:
        # Valid JSON but missing required 'user_input' field
        await consumer.process_message(json.dumps({"unexpected_field": "value"}))
        mock_graph.ainvoke.assert_not_called()

    async def test_graph_error_produces_error_response(
        self, consumer: MQConsumer, mock_graph: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True
        mock_graph.ainvoke.side_effect = RuntimeError("graph exploded")
        await consumer.process_message(make_request_json(request_id="err-id"))
        # Should still publish a response (error response)
        mock_redis.lpush.assert_called_once()
        _, payload = mock_redis.lpush.call_args[0]
        response = ResponseMessage.model_validate_json(payload)
        assert response.error is not None
        assert "graph exploded" in response.error

    async def test_graph_error_response_has_fallback_message(
        self, consumer: MQConsumer, mock_graph: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True
        mock_graph.ainvoke.side_effect = ValueError("tool not found")
        await consumer.process_message(make_request_json())
        _, payload = mock_redis.lpush.call_args[0]
        response = ResponseMessage.model_validate_json(payload)
        assert response.final_response  # Non-empty even on error

    async def test_graph_returns_none_final_response_uses_fallback(
        self, consumer: MQConsumer, mock_graph: AsyncMock, mock_redis: AsyncMock
    ) -> None:
        mock_redis.set.return_value = True
        mock_graph.ainvoke.return_value = {"error": "synthesis failed", "final_response": None}
        await consumer.process_message(make_request_json())
        _, payload = mock_redis.lpush.call_args[0]
        response = ResponseMessage.model_validate_json(payload)
        assert "synthesis failed" in response.final_response


# ---------------------------------------------------------------------------
# MQConsumer — lifecycle
# ---------------------------------------------------------------------------


class TestMQConsumerLifecycle:
    async def test_stop_sets_running_false(
        self, consumer: MQConsumer
    ) -> None:
        consumer._running = True
        await consumer.stop()
        assert consumer._running is False

    async def test_context_manager_connects(
        self, mock_redis: AsyncMock, mock_graph: AsyncMock
    ) -> None:
        consumer = MQConsumer(graph=mock_graph, _redis_client=mock_redis)
        async with consumer as c:
            assert c._redis is mock_redis

    async def test_context_manager_disconnects(
        self, mock_redis: AsyncMock, mock_graph: AsyncMock
    ) -> None:
        consumer = MQConsumer(graph=mock_graph, _redis_client=mock_redis)
        async with consumer:
            pass
        mock_redis.aclose.assert_called_once()

    async def test_not_connected_raises_on_process(
        self, mock_graph: AsyncMock
    ) -> None:
        consumer = MQConsumer(graph=mock_graph)  # No client, not connected
        with pytest.raises(MessageQueueError, match="not connected"):
            await consumer.process_message(make_request_json())
