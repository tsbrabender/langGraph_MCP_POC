# LangGraph + FastMCP + Ollama Agent

A complete, local-first AI agent that uses **LangGraph** for workflow orchestration, **FastMCP** for tool exposure, and a local **Ollama** LLM for both tool selection and response synthesis. Everything runs on a single machine — no cloud accounts, no API keys.

---

## What it does

The agent accepts natural-language requests and responds with grounded, tool-backed answers:

```
User: "list the files in the sandbox"
Agent: "I found 2 items: hello.txt (53 B) and example.json (187 B)."

User: "read hello.txt"
Agent: "The file says: Hello from the sandbox! ..."

User: "find all *.json files"
Agent: "I found 1 match: example.json"
```

All answers are backed by real tool output — the LLM synthesises language but cannot hallucinate file contents.

---

## Architecture

### Two-pass LLM pattern

```
User Input
    │
    ▼
[ingest_user_input]       Validate and stamp request metadata
    │
    ▼
[classify_intent]         Hybrid: keyword regex → try to extract tool + args
    │  ┌── full match ──────────────────────────────────────────┐
    │  │                                                         │
    ▼  ▼                                                         │
[llm_tool_selection]      Ollama Pass 1: choose tool + generate args
    │                                                            │
    └────────────────────────────────────────────────────────────┘
                          │
                          ▼
              [mcp_tool_invocation]    Dispatch to the selected MCP tool
                          │
                          ▼
              [llm_response_synthesis] Ollama Pass 2: synthesise natural-language answer
                          │
                          ▼
              [finalize_response]      Stamp timestamps, append conversation history, persist to SQLite
                          │
                          ▼
                    Final Response
```

**Hybrid routing** — the `classify_intent` node tries to extract the tool and arguments from the user's input using fast regular expressions. On a full match it routes directly to `mcp_tool_invocation`, skipping the LLM call entirely. On a partial match it passes the intent as a hint to the LLM. Unrecognised inputs fall through to the LLM unconditionally.

### MQ mode (optional)

When `MQ_ENABLED=true` the UI and the graph are decoupled via Redis:

```
FastAPI UI  →  LPUSH requests  →  Redis  →  BRPOP  →  MQ Consumer
                                                              │
                                                              ▼
                                                         LangGraph
                                                              │
                                                              ▼
FastAPI UI  ←  BRPOP response:<id>  ←  Redis  ←  LPUSH  MQ Consumer
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | |
| [Ollama](https://ollama.ai) | any | Must be running before starting the agent |
| Redis | 7+ | **Only** needed when `MQ_ENABLED=true` |
| Docker + Compose | any | Optional — only for the full containerised stack |

Pull a model before first run:

```bash
ollama pull llama3.2
```

---

## Quick start — local (direct mode, no Redis)

```bash
# 1. Clone and install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env            # edit if Ollama runs on a non-default port

# 3. Start Ollama (separate terminal or background process)
ollama serve

# 4. Start the web UI
uvicorn app.ui.api:app --reload

# 5. Open http://localhost:8000 in a browser
```

Try: `list files`, `read hello.txt`, `find *.json`, `summarise example.json`

---

## Quick start — Docker (MQ mode, full stack)

```bash
# Build and start everything
docker compose up --build

# UI is available at http://localhost:8080
# Ollama must be running on the host — Docker connects via host.docker.internal
```

Services started by `docker compose up`:

| Service | What it does |
|---|---|
| `redis` | Message queue backend |
| `consumer` | BRPOPs from the request queue, runs LangGraph, publishes responses |
| `ui` | FastAPI server — serves the web UI and `/api/*` endpoints |

---

## Environment variables

All settings live in `.env` (copy from `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name as returned by `ollama list` |
| `MCP_SERVER_HOST` | `0.0.0.0` | FastMCP server bind address |
| `MCP_SERVER_PORT` | `8000` | FastMCP server port |
| `SQLITE_DB_PATH` | `data/workflow.db` | SQLite database file path |
| `MQ_ENABLED` | `false` | Set to `true` to activate Redis-backed async mode |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `MQ_REQUEST_QUEUE` | `requests` | Redis list name for incoming requests |
| `MQ_RESPONSE_QUEUE` | `responses` | Redis list name for outgoing responses |
| `SANDBOX_ROOT` | `./sandbox` | Root directory the agent is allowed to read |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Running tests

```bash
# All tests (244 total, no live Ollama or Redis needed)
pytest

# Unit tests only
pytest tests/unit/

# Integration tests (real file I/O, mocked LLM)
pytest tests/integration/

# Specific module
pytest tests/unit/test_tools.py -v
```

Test counts by module:

| File | Tests | What is covered |
|---|---|---|
| `test_tools.py` | 49 | All 5 MCP tools + sandbox path enforcement |
| `test_llm.py` | 34 | ToolSelector, ResponseSynthesizer, ToolRegistry |
| `test_graph_nodes.py` | 42 | All 6 LangGraph nodes + 3 routing functions |
| `test_graph.py` (integration) | 18 | Full `ainvoke()` round-trip, both graph variants |
| `test_mq.py` | 40 | MQ schemas, producer, consumer, idempotency |
| `test_ui.py` | 36 | FastAPI endpoints, direct + MQ modes, history |
| `test_db.py` | 25 | SQLiteClient, WorkflowRun, migrations, node persistence |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI (HTML) |
| `POST` | `/api/chat` | Submit a request; returns `ChatResponse` |
| `GET` | `/api/health` | Liveness check — returns mode and MQ status |
| `GET` | `/api/history?limit=N` | Recent workflow runs from SQLite |

### `POST /api/chat`

**Request:**
```json
{ "user_input": "list the files" }
```

**Response:**
```json
{
  "request_id": "550e8400-...",
  "user_input": "list the files",
  "selected_tool": "list_files",
  "tool_output": { "count": 2, "entries": [...], "directory": "." },
  "final_response": "I found 2 files in the sandbox: hello.txt and example.json.",
  "error": null,
  "mode": "direct"
}
```

---

## Available MCP tools

| Tool | What it does | Key limits |
|---|---|---|
| `list_files` | List directory contents | Sorted: dirs first, then files |
| `read_file` | Read a file's text content | Max 1 MB; truncates silently |
| `search_files` | Glob-pattern file search | Max 500 results |
| `extract_metadata` | Size, timestamps, permissions | Works on files and directories |
| `summarise_file` | LLM summary of a file | Reads up to 4 000 chars |

All tools are sandboxed to `SANDBOX_ROOT`. Path traversal (`../../etc/passwd`) raises `SandboxViolationError`.

---

## Project structure

```
app/
  mcp_server/
    tools/              One file per MCP tool + shared sandbox helper
    server.py           FastMCP server entry point
  graph/
    nodes/              One file per LangGraph node (pure functions)
    edges/routing.py    Conditional edge logic
    state.py            TypedDict GraphState (total=False)
    graph.py            build_llm_graph() and build_hybrid_graph()
  llm/
    ollama_client.py    Async Ollama wrapper
    tool_selector.py    Pass 1 — structured JSON tool call from LLM
    response_synthesizer.py  Pass 2 — natural-language answer from LLM
    tool_registry.py    Connects LLM layer to MCP tool schemas
  services/
    mcp_executor.py     Dispatches tool calls (graph → tools, no direct import)
    db/
      sqlite_client.py  Async SQLite with auto-migration
      models.py         WorkflowRun Pydantic model
      migrations/       001_initial.sql (applied on connect)
    mq/
      schemas.py        RequestMessage / ResponseMessage
      producer.py       Publish requests, poll responses (Redis LPUSH/BRPOP)
      consumer.py       BRPOP loop with idempotency (Redis SET NX)
      runner.py         python -m app.services.mq.runner entry point
  ui/
    api.py              FastAPI app (create_app factory for testing)
    static/index.html   Single-page web UI
  utils/
    config.py           Settings from .env via pydantic-settings
    logging.py          structlog JSON logging
    errors.py           Typed exception hierarchy
tests/
  unit/                 Pure tests — no Ollama, Redis, or file system
  integration/          End-to-end graph tests with real file I/O
  fixtures/sandbox/     Static files used by integration tests
sandbox/                Default agent working directory (gitignored data)
```

---

## How to extend

See [CLAUDE.md](CLAUDE.md) for the full specification including:

- **Add a new MCP tool** — one file in `app/mcp_server/tools/`, register in `server.py` and `tool_registry.py`
- **Add a new LangGraph node** — implement `make_node(dep) -> Callable[[GraphState], dict]`, wire into `graph.py`
- **Add a new database table** — new migration in `app/services/db/migrations/`, query helpers in `sqlite_client.py`
- **Add a new message queue consumer** — handler in `consumer.py`, ensure idempotency

---

## Operational notes

- **Ollama must be running** before the UI or consumer starts. Check with `ollama list`.
- **SQLite** is created automatically at `SQLITE_DB_PATH` on first connect. No setup needed.
- **Redis** is only required when `MQ_ENABLED=true`. The system runs fully without it.
- **Sandbox isolation** is enforced at the Python level (not OS-level). Do not place sensitive files inside `SANDBOX_ROOT`.
- **MQ consumer** (`python -m app.services.mq.runner`) is a separate long-running process. The UI publishes requests; the consumer processes them. Both must be running in MQ mode.
- **Structured logs** are emitted as JSON on stdout via `structlog`. Set `LOG_LEVEL=DEBUG` to see full LLM prompt/response traces.
