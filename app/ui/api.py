"""FastAPI application — the UI layer.

Two modes, controlled by MQ_ENABLED in settings:

  direct  (MQ_ENABLED=false, default)
      POST /api/chat  →  graph.ainvoke()  →  ChatResponse

  mq  (MQ_ENABLED=true)
      POST /api/chat  →  MQProducer.publish_request()  →  BRPOP response key
                     →  ChatResponse

The UI never imports MCP tools or the Ollama client directly.

Additional endpoints:
  GET /api/health    — liveness check with mode information
  GET /api/history   — recent workflow runs from SQLite (empty list when DB disabled)

To start the server:
    uvicorn app.ui.api:app --reload

For testability, create an isolated app instance via create_app():
    from app.ui.api import create_app
    app = create_app(graph=mock_graph)
"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.utils.config import get_settings
from app.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    user_input: str
    model: str | None = None  # Ollama model name; falls back to OLLAMA_MODEL env / config default


class ChatResponse(BaseModel):
    request_id: str
    user_input: str
    selected_tool: str | None = None
    tool_output: Any = None
    final_response: str
    error: str | None = None
    mode: str  # "direct" | "mq"


class ModelsResponse(BaseModel):
    models: list[str]
    default: str


class HealthResponse(BaseModel):
    status: str
    mq_enabled: bool
    mode: str


class HistoryResponse(BaseModel):
    runs: list[Any]
    total: int


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    graph: Any | None = None,
    producer: Any | None = None,
    db_client: Any | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        graph:     Pre-built compiled LangGraph. When None the lifespan builds one
                   from OllamaClient + MCPExecutor (requires Ollama running).
        producer:  Pre-built MQProducer. When None and MQ_ENABLED=true the
                   lifespan creates one (requires Redis running).
        db_client: Pre-built SQLiteClient for history persistence. When None and
                   a graph is being built, the lifespan creates and connects one.

    The optional parameters exist solely for testing — pass mock objects so
    no real services are needed.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        configure_logging(settings.log_level)

        # --- SQLite ---
        if app.state.db_client is None and app.state.graph is None:
            # Real startup: build the db client before the graph so we can thread it in.
            from app.services.db.sqlite_client import SQLiteClient

            _db = SQLiteClient()
            await _db.connect()
            app.state.db_client = _db
            log.info("ui_db_connected")

        # --- Cache client ---
        if app.state.db_client is not None and not hasattr(app.state, "cache_client"):
            from app.services.cache.sqlite_cache import SQLiteCacheClient

            app.state.cache_client = SQLiteCacheClient(app.state.db_client)
            log.info("ui_cache_client_ready")

        # --- Graph ---
        if app.state.graph is None:
            from app.graph.graph import build_context_aware_graph
            from app.llm.ollama_client import OllamaClient
            from app.llm.response_synthesizer import ResponseSynthesizer
            from app.llm.tool_registry import build_tool_definitions
            from app.llm.tool_selector import ToolSelector
            from app.services.mcp_executor import MCPExecutor

            llm = OllamaClient()
            selector = ToolSelector(llm, build_tool_definitions())
            cache_client = getattr(app.state, "cache_client", None)
            executor = MCPExecutor(
                sandbox_root=Path(settings.sandbox_root).resolve(),
                llm_client=llm,
                cache_client=cache_client,
            )
            synthesizer = ResponseSynthesizer(llm)
            app.state.graph = build_context_aware_graph(
                selector,
                executor,
                synthesizer,
                cache_client=cache_client,
                db_client=app.state.db_client,
            )
            log.info("ui_graph_built", mode="context_aware")

        # --- MQ producer ---
        if settings.mq_enabled and app.state.producer is None:
            from app.services.mq.producer import MQProducer

            app.state.producer = MQProducer()
            await app.state.producer.connect()
            log.info("ui_mq_producer_connected")

        app.state.mq_enabled = settings.mq_enabled
        yield

        if app.state.producer is not None:
            await app.state.producer.disconnect()
            log.info("ui_mq_producer_disconnected")

        if app.state.db_client is not None:
            await app.state.db_client.disconnect()
            log.info("ui_db_disconnected")

    _app = FastAPI(
        title="LangGraph MCP Agent",
        description="Local-first file-system agent powered by LangGraph, FastMCP, and Ollama.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Seed state so lifespan can detect pre-injected deps.
    _app.state.graph = graph
    _app.state.producer = producer
    _app.state.db_client = db_client
    _app.state.mq_enabled = False

    # ------------------------------------------------------------------ #
    # Routes                                                               #
    # ------------------------------------------------------------------ #

    @_app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html)

    @_app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        mq_on = bool(_app.state.mq_enabled)
        return HealthResponse(
            status="ok",
            mq_enabled=mq_on,
            mode="mq" if mq_on else "direct",
        )

    @_app.get("/api/models", response_model=ModelsResponse)
    async def models() -> ModelsResponse:
        """Return the list of Ollama models available on this machine."""
        import ollama

        settings = get_settings()
        try:
            client = ollama.AsyncClient(host=settings.ollama_base_url)
            result = await client.list()
            names: list[str] = sorted(
                m["model"] for m in result.get("models", []) if m.get("model")
            )
        except Exception as exc:
            log.error("ui_models_fetch_failed", error=str(exc))
            raise HTTPException(
                status_code=503, detail=f"Could not reach Ollama: {exc}"
            ) from exc

        return ModelsResponse(models=names, default=settings.ollama_model)

    @_app.post("/api/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        user_input = body.user_input.strip()
        if not user_input:
            raise HTTPException(status_code=422, detail="user_input must not be empty.")

        if _app.state.mq_enabled and _app.state.producer is not None:
            return await _chat_mq(_app, user_input)
        return await _chat_direct(_app, user_input, model=body.model)

    @_app.get("/api/history", response_model=HistoryResponse)
    async def history(limit: int = Query(default=20, ge=1, le=200)) -> HistoryResponse:
        """Return recent workflow runs from SQLite, newest first.

        Returns an empty list when the DB client is not configured.
        """
        db = _app.state.db_client
        if db is None:
            return HistoryResponse(runs=[], total=0)
        try:
            runs = await db.list_runs(limit=limit)
            return HistoryResponse(
                runs=[r.model_dump() for r in runs],
                total=len(runs),
            )
        except Exception as exc:
            log.error("ui_history_error", error=str(exc))
            raise HTTPException(status_code=500, detail=f"History query failed: {exc}") from exc

    return _app


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------


async def _chat_direct(app: FastAPI, user_input: str, model: str | None = None) -> ChatResponse:
    request_id = str(uuid.uuid4())
    initial_state: dict[str, Any] = {
        "user_input": user_input,
        "metadata": {"request_id": request_id},
    }
    if model:
        initial_state["model"] = model
    try:
        final_state: dict[str, Any] = await app.state.graph.ainvoke(initial_state)
    except Exception as exc:
        log.error("ui_direct_graph_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {exc}") from exc

    return ChatResponse(
        request_id=request_id,
        user_input=user_input,
        selected_tool=final_state.get("selected_tool"),
        tool_output=final_state.get("tool_output"),
        final_response=final_state.get("final_response") or "No response generated.",
        error=final_state.get("error"),
        mode="direct",
    )


async def _chat_mq(app: FastAPI, user_input: str) -> ChatResponse:
    producer = app.state.producer
    msg = await producer.publish_request(user_input)
    response = await producer.get_response(msg.request_id, timeout=30.0)

    if response is None:
        raise HTTPException(
            status_code=504,
            detail="The agent did not respond within 30 seconds. Is the consumer running?",
        )

    return ChatResponse(
        request_id=response.request_id,
        user_input=user_input,
        selected_tool=response.selected_tool,
        tool_output=response.tool_output,
        final_response=response.final_response,
        error=response.error,
        mode="mq",
    )


# ---------------------------------------------------------------------------
# Default application instance (used by uvicorn)
# ---------------------------------------------------------------------------

app = create_app()
