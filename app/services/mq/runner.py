"""Entry point for the MQ consumer worker process.

Run with:
    python -m app.services.mq.runner

The consumer builds its own LangGraph, connects to Redis, and processes
requests until interrupted with Ctrl-C.  MQ_ENABLED must be true.
"""

import asyncio
import signal
from pathlib import Path

from app.graph.graph import build_llm_graph
from app.llm.ollama_client import OllamaClient
from app.llm.response_synthesizer import ResponseSynthesizer
from app.llm.tool_registry import build_tool_definitions
from app.llm.tool_selector import ToolSelector
from app.services.mcp_executor import MCPExecutor
from app.services.mq.consumer import MQConsumer
from app.utils.config import get_settings
from app.utils.logging import configure_logging, get_logger


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger(__name__)

    if not settings.mq_enabled:
        log.error(
            "mq_disabled",
            message="Set MQ_ENABLED=true in .env to start the consumer.",
        )
        return

    llm = OllamaClient()
    selector = ToolSelector(llm, build_tool_definitions())
    executor = MCPExecutor(
        sandbox_root=Path(settings.sandbox_root).resolve(),
        llm_client=llm,
    )
    synthesizer = ResponseSynthesizer(llm)
    graph = build_llm_graph(selector, executor, synthesizer)

    consumer = MQConsumer(graph=graph)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(consumer.stop()))

    log.info("consumer_runner_starting", queue=settings.mq_request_queue)
    await consumer.start()


if __name__ == "__main__":
    asyncio.run(main())
