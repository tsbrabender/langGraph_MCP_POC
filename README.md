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

When a user asks about a configured topic (e.g. "tell me about dyslexia", "what is adhd?"), the agent automatically fetches reference material from pre-configured URLs, caches it in SQLite, and provides it to the LLM as grounding context before generating an answer. Cache TTLs are configurable per topic; subsequent requests on the same topic are served from cache without a network call.

The web UI includes a **model selection dropdown** populated dynamically from your local Ollama installation (`GET /api/models`). The selected model is sent with each request and used for both the tool-selection pass and the response-synthesis pass. The selection is remembered in `localStorage` across page refreshes. The `OLLAMA_MODEL` environment variable sets the startup default used when no model is explicitly chosen.

---

## Architecture

### Default graph — context-aware hybrid

The default graph (`build_context_aware_graph`) has 8 nodes. The LLM is only called for tool selection (Pass 1) and response synthesis (Pass 2); all other nodes are deterministic.

```
User Input
    │
    ▼
[ingest_user_input]     Validate and stamp request metadata
    │
    ▼
[detect_topic]          Keyword scan against topic_map.yaml
    │                   (e.g. "dyslexia", "adhd", "anxiety")
    │  ┌── topic found ──────────────────────────────────────┐
    │  │                                                      │
    ▼  ▼                                                      │
[retrieve_context]      Cache-first fetch of topic URLs       │
    │                   → writes context_documents to state   │
    └──────────────────────────────────────────────────────── ┘
                        │
                        ▼
            [classify_intent]   Keyword regex → extract tool + args
                │  ┌── full match ─────────────────────────────────┐
                │  │                                                │
                ▼  ▼                                                │
        [llm_tool_selection]    Ollama Pass 1: choose tool + args   │
            │                                                       │
            └───────────────────────────────────────────────────────┘
                        │
                        ▼
            [mcp_tool_invocation]    Dispatch to the selected MCP tool
                        │
                        ▼
            [llm_response_synthesis] Ollama Pass 2: synthesise answer
                                     (enriched with context_documents when present)
                        │
                        ▼
            [finalize_response]      Stamp timestamps, persist to SQLite
                        │
                        ▼
                  Final Response
```

**Topic detection** — `detect_topic` scans user input against keyword lists in `app/mcp_server/resources/topic_map.yaml`. On a match it sets `state["topic"]`; `retrieve_context` then fetches the configured URLs (cache-first, falling back to live fetch) and populates `state["context_documents"]`. The synthesizer receives these documents as grounding material.

**Hybrid routing** — `classify_intent` applies fast regex patterns to extract a tool name and arguments. On a full match it routes directly to `mcp_tool_invocation`, skipping the LLM. On a partial match it passes the tool name as a hint to `llm_tool_selection`. Unrecognised inputs fall through to full LLM selection.

Two simpler graph variants are also available: `build_llm_graph()` (fully LLM-driven, no keyword classification) and `build_hybrid_graph()` (adds `classify_intent` but no topic detection).

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

## Virtual environment setup

A virtual environment keeps the project's dependencies isolated from your system Python. Complete this once before the quick-start steps below.

**Windows (PowerShell)**
```powershell
# Create the environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1

# If PowerShell blocks the script, run this first (once per machine):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Once the environment is active your prompt shows `(.venv)`. Install all dependencies into it:

```bash
pip install -e ".[dev]"
```

**Deactivating and reactivating**

```bash
deactivate               # leave the environment
source .venv/bin/activate  # re-enter (macOS/Linux)
.venv\Scripts\Activate.ps1 # re-enter (Windows)
```

> **VS Code** — after creating the environment, press `Ctrl+Shift+P` → **Python: Select Interpreter** and choose the `.venv` entry. The IDE will then use the correct interpreter for IntelliSense, linting, and the integrated terminal.

---

## Quick start — local (direct mode, no Redis)

```bash
# 1. Activate your virtual environment (if not already active)
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\Activate.ps1    # Windows

# 2. Install dependencies (first time only, or after pulling new changes)
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env            # edit if Ollama runs on a non-default port

# 4. Start Ollama (separate terminal or background process)
ollama serve

# 5. Start the web UI
uvicorn app.ui.api:app --reload

# 6. Open http://localhost:8000 in a browser
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
| `OLLAMA_MODEL` | `llama3.2` | Default model used when the UI sends no model override |
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
| `test_tools.py` | 49 | 5 file-system MCP tools + sandbox path enforcement |
| `test_llm.py` | 34 | ToolSelector, ResponseSynthesizer, ToolRegistry |
| `test_graph_nodes.py` | 42 | 6 original LangGraph nodes + 3 original routing functions |
| `test_graph.py` (integration) | 18 | Full `ainvoke()` round-trip, llm and hybrid graph variants |
| `test_mq.py` | 40 | MQ schemas, producer, consumer, idempotency |
| `test_ui.py` | 36 | FastAPI endpoints, direct + MQ modes, history |
| `test_db.py` | 25 | SQLiteClient, WorkflowRun, migrations, node persistence |

> **Tests pending** — the following additions do not yet have test coverage: 4 context-retrieval MCP tools (`get_topic_resources`, `fetch_web_resource`, `get_cached_resource`, `refresh_cache`), 2 new graph nodes (`detect_topic`, `retrieve_context`), the cache service (`SQLiteCacheClient`), `TopicMap` loader, and `build_context_aware_graph()`.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI (HTML) |
| `GET` | `/api/models` | List of Ollama models available on this machine |
| `POST` | `/api/chat` | Submit a request; returns `ChatResponse` |
| `GET` | `/api/health` | Liveness check — returns mode and MQ status |
| `GET` | `/api/history?limit=N` | Recent workflow runs from SQLite |

### `GET /api/models`

```json
{
  "models": ["llama3.2:latest", "mistral:7b", "phi3:mini"],
  "default": "llama3.2"
}
```

Returns every model currently available in the local Ollama installation, sorted alphabetically, plus the server-configured default (`OLLAMA_MODEL`). Returns `503` if Ollama is unreachable.

### `POST /api/chat`

**Request:**
```json
{
  "user_input": "list the files",
  "model": "mistral:7b"
}
```

`model` is optional. When omitted the server uses the `OLLAMA_MODEL` default.

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

### File-system tools (sandboxed to `SANDBOX_ROOT`)

| Tool | What it does | Key limits |
|---|---|---|
| `list_files` | List directory contents | Sorted: dirs first, then files |
| `read_file` | Read a file's text content | Max 1 MB; truncates silently |
| `search_files` | Glob-pattern file search | Max 500 results |
| `extract_metadata` | Size, timestamps, permissions | Works on files and directories |
| `summarize_file` | LLM summary of a file | Reads up to 4 000 chars |

Path traversal (`../../etc/passwd`) raises `SandboxViolationError`.

### Context-retrieval tools

| Tool | What it does | Key limits |
|---|---|---|
| `get_topic_resources` | Returns configured URLs for a topic | Reads `topic_map.yaml`; empty list when unknown |
| `fetch_web_resource` | Fetches and normalizes web page content | 512 KB download cap; output truncated to 8 000 chars; 15 s timeout |
| `get_cached_resource` | Returns cached content when TTL is still valid | Returns `hit=False` on miss or expiry |
| `refresh_cache` | Forces a live fetch and overwrites the cache entry | Bypasses TTL; use when user requests fresh data |

Context-retrieval tools are primarily orchestrated by the `retrieve_context` node and can also be selected directly by the LLM for explicit user requests.

---

## Project structure

```
app/
  mcp_server/
    tools/              One file per MCP tool (9 tools + shared sandbox helper)
    resources/
      topic_map.yaml    Topic keyword lists, resource URLs, and per-topic TTL settings
    server.py           FastMCP server entry point
  graph/
    nodes/              One file per LangGraph node (8 nodes total)
    edges/routing.py    Conditional edge logic (5 routing functions)
    state.py            TypedDict GraphState (total=False)
    graph.py            build_llm_graph(), build_hybrid_graph(), build_context_aware_graph()
  llm/
    ollama_client.py    Async Ollama wrapper (per-call model override)
    tool_selector.py    Pass 1 — structured JSON tool call from LLM
    response_synthesizer.py  Pass 2 — natural-language answer, enriched with context_documents
    tool_registry.py    Connects LLM layer to all 9 MCP tool schemas
  services/
    mcp_executor.py     Dispatches tool calls; injects llm_client and cache_client
    cache/
      cache_client.py   CacheClient interface and ResourceCacheEntry data class
      sqlite_cache.py   SQLite-backed cache (get / set / delete)
    db/
      sqlite_client.py  Async SQLite with auto-migration
      models.py         WorkflowRun Pydantic model
      migrations/       001_initial.sql, 002_resource_cache.sql
    mq/
      schemas.py        RequestMessage / ResponseMessage
      producer.py       Publish requests, poll responses (Redis LPUSH/BRPOP)
      consumer.py       BRPOP loop with idempotency (Redis SET NX)
      runner.py         python -m app.services.mq.runner entry point
  ui/
    api.py              FastAPI app (create_app factory for testing)
    static/index.html   Single-page web UI with model selection dropdown
  utils/
    config.py           Settings from .env via pydantic-settings
    topic_config.py     TopicMap loader and Pydantic validator for topic_map.yaml
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
