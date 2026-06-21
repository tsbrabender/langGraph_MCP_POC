# CLAUDE.md — Project Reference

This file defines the project's tech stack, architecture, coding standards, and operational assumptions. It serves as the primary reference for any developer or AI assistant working in this codebase.

---

## Tech Stack

### Core Technologies

| Technology | Role |
|---|---|
| **Python 3.12+** | Primary language |
| **LangGraph** | Workflow orchestration and stateful agent graphs |
| **FastMCP** | MCP server framework for exposing tools, resources, and prompts |
| **Local Ollama LLM** | Determines which MCP tool to call and generates natural-language responses |
| **MCP Client** | Client-side integration for calling external MCP-compliant tools |
| **SQLite** | Local persistent storage and workflow checkpoints |
| **Message Queue** (Redis / NATS / RabbitMQ) | Async task dispatch, fan-out, background workers |
| **Pydantic** | Schema validation and typed I/O contracts |
| **AsyncIO** | Concurrency model throughout the stack |
| **Docker** | Reproducible local containerized runtime |

### Optional

- **Internet-based MCP tools** — external data access (web search, APIs) invoked through MCP tool calls when network is available.

### Why This Stack

- **Local-first execution** — SQLite, Ollama, and FastMCP require no cloud services; the full system runs on a single machine.
- **LLM-driven tool selection** — the local Ollama LLM determines which MCP tool to invoke based on user intent and current graph state.
- **Deterministic workflows** — LangGraph's graph-based execution makes control flow explicit, auditable, and reproducible.
- **Natural-language output** — after tool execution, the LLM synthesizes results into a human-friendly response.
- **Modular tool exposure** — MCP tools remain isolated, typed, and independently testable.
- **Simple deployment** — Docker Compose runs the full local stack; no cloud infrastructure required.

---

## Architecture

### High-Level Workflow (Default: Context-Aware Hybrid Graph)

The default graph (`build_context_aware_graph`) runs 8 nodes. The Ollama LLM is invoked in two passes only; all other nodes are deterministic.

```
User Input
    │
    ▼
[ingest_user_input]     Validate and stamp request metadata
    │
    ▼
[detect_topic]          Keyword scan against topic_map.yaml
    │  ┌── topic found ──────────────────────────────────────────┐
    │  │                                                          │
    ▼  ▼                                                          │
[retrieve_context]      Cache-first fetch of topic resource URLs  │
    │                   → populates state["context_documents"]    │
    └──────────────────────────────────────────────────────────── ┘
                        │
                        ▼
            [classify_intent]   Keyword regex → try to extract tool + args
                │  ┌── full match ─────────────────────────────────────┐
                │  │                                                    │
                ▼  ▼                                                    │
        [llm_tool_selection]    Ollama Pass 1: choose tool + args       │
            │                                                           │
            └───────────────────────────────────────────────────────────┘
                        │
                        ▼
            [mcp_tool_invocation]    Execute the selected MCP tool
                        │
                        ▼
            [llm_response_synthesis] Ollama Pass 2: synthesise response
                                     (context_documents injected into prompt when present)
                        │
                        ▼
            [finalize_response]      Stamp timestamps, persist to SQLite
                        │
                        ▼
                  Final Response
```

**Pass 1 — Tool Selection:** The LLM receives user input and all 10 registered MCP tool schemas (read live from `ToolRegistry`). Returns a structured JSON tool call validated against the tool's Pydantic schema before execution.

**Pass 2 — Response Synthesis:** The LLM receives the original user input, MCP tool output, and — when a topic was detected — the `context_documents` from `retrieve_context` as grounding material.

Two simpler variants exist: `build_llm_graph()` (fully LLM-driven, 5 nodes) and `build_hybrid_graph()` (adds `classify_intent`, 6 nodes). Both share the same LangGraph node implementations.

---

### Recommended Architecture: Hybrid Routing

While the two-pass LLM pattern is flexible, a **hybrid approach** is the default in this project:

- **Keyword regex for intent classification** — the `classify_intent` node runs fast regular-expression patterns against the user input to extract a tool name and arguments without an LLM call.
- **Rule-based routing for tool selection** — LangGraph edges deterministically route to `mcp_tool_invocation` on a full keyword match, skipping the LLM entirely. On a partial match the extracted tool name is passed as a hint to the LLM.
- **LLM only for final response synthesis** — reduces total LLM calls, eliminates tool hallucination, and improves latency.

This hybrid model preserves the natural-language flexibility of LLM-driven workflows while avoiding the reliability issues of fully LLM-driven tool selection.

---

### Directory Structure

```
/project-root
    /app
        /mcp_server
            tool_spec.py        # ToolSpec dataclass — contract between tools and the loader
            tool_loader.py      # discover_categories(), discover_tools(), discover_all_tools()
            /tools
                /_sandbox.py    # resolve_safe_path() — shared path-traversal guard
                /file_ops/      # Category: sandboxed file-system tools
                    /list_files/        list_files.tool + schemas + __init__
                    /read_file/         read_file.tool + schemas + __init__
                    /search_files/      search_files.tool + schemas + __init__
                    /extract_metadata/  extract_metadata.tool + schemas + __init__
                /summarization/ # Category: LLM-backed summarisation tools
                    /summarize_file/    summarize_file.tool + schemas + __init__
                    /summarize_text/    summarize_text.tool + schemas + __init__
                /context_retrieval/  # Category: topic-aware web-resource tools
                    /get_topic_resources/
                    /fetch_web_resource/
                    /get_cached_resource/
                    /refresh_cache/
            /resources
                topic_map.yaml  # Topic → keyword list + resource URLs + TTL
            /prompts            # MCP prompt templates (placeholder)
            server.py           # FastMCP server — registers all tools dynamically
        /graph
            /nodes              # LangGraph node functions (8 nodes, make_node factory)
            /edges
                routing.py      # Conditional edge logic (5 routing functions)
            state.py            # GraphState TypedDict (total=False)
            graph.py            # build_llm_graph(), build_hybrid_graph(),
                                #   build_context_aware_graph()
        /llm
            ollama_client.py        # Async Ollama wrapper (per-call model override)
            tool_selector.py        # Pass 1 — LLM tool selection; reads live ToolRegistry
            response_synthesizer.py # Pass 2 — grounded synthesis with optional context_documents
            tool_registry.py        # ToolRegistry class — thread-safe, hot-reloadable
        /services
            mcp_executor.py         # Dynamic dispatch via inspect.signature + ToolRegistry
            /cache
                cache_client.py     # CacheClient interface + ResourceCacheEntry data class
                sqlite_cache.py     # SQLite-backed get / set / delete
            /db
                sqlite_client.py    # Async SQLite connection and query helpers
                models.py           # WorkflowRun Pydantic model
                /migrations
                    001_initial.sql         # workflow_runs table
                    002_resource_cache.sql  # resource_cache table
            /mq
                producer.py         # Message queue publish interface
                consumer.py         # Message queue subscribe/consume logic
                runner.py           # python -m app.services.mq.runner entry point
                schemas.py          # RequestMessage / ResponseMessage
        /ui
            api.py              # FastAPI app (create_app factory); POST /api/tools/refresh
            /static
                index.html      # Single-page web UI: model dropdown, Tool Refresh button
        /utils
            config.py           # pydantic-settings — reads from .env
            topic_config.py     # TopicMap loader and validator for topic_map.yaml
            logging.py          # structlog JSON logging setup
            errors.py           # Typed exception hierarchy
    /tests
        /unit                   # Pure tests — no Ollama, Redis, or file system
        /integration            # End-to-end graph and API tests with real file I/O
    CLAUDE.md
    pyproject.toml
    README.md
```

---

### Architectural Principles

- **LLM-driven tool selection** — the local Ollama LLM decides which MCP tool to call based on user intent and tool schemas.
- **Two-pass LLM workflow** — Pass 1 selects the tool; Pass 2 synthesizes the natural-language response from tool output.
- **Per-request model selection** — the UI sends an optional `model` field with each request. It flows through `GraphState["model"]` and is passed as an override to every `OllamaClient.chat()` call in that request, overriding the `OLLAMA_MODEL` default without restarting the server.
- **Separation of concerns** — LLM logic lives in `/app/llm/`, MCP tools in `/app/mcp_server/`, graph orchestration in `/app/graph/`. No cross-layer imports.
- **LangGraph nodes are pure** — nodes orchestrate state transitions only; business logic and LLM calls are delegated to injected service clients.
- **SQLite for durable state** — workflow checkpoints, audit logs, and tool output history.
- **MQ for async work** — optional; used when background tasks or fan-out patterns are needed.
- **Local-first** — the system runs fully offline unless a tool explicitly requires internet access.

---

## Code Standards

### General

- Follow **PEP 8** for style and **PEP 484** for type hints.
- Use **type hints on all function signatures** — parameters and return types.
- Write **docstrings on all public functions and classes** — describe purpose, parameters, and return value.
- Favor **pure functions** — given the same inputs, return the same outputs with no hidden state.
- **No global mutable state** — pass dependencies explicitly via function arguments or constructor injection.
- Use **Pydantic models** for all input/output schemas.
- Favor **composition over inheritance**.
- Log at all workflow entry points, node transitions, LLM calls, tool invocations, and error boundaries.

### LLM Tool Selection (`app/llm/tool_selector.py`)

- Tool selection prompts must be **deterministic and schema-bound** — the LLM is given the exact Pydantic schemas of available tools.
- LLM output **must be validated** against the target tool's input schema before the MCP client is invoked.
- If validation fails, the node must raise a typed error rather than attempting execution with invalid arguments.
- Tool selection logic is isolated in `tool_selector.py` and must be independently testable with mock LLM responses.

### LLM Response Synthesis (`app/llm/response_synthesizer.py`)

- The LLM must receive:
  - original user input
  - MCP tool output (verbatim structured result)
  - relevant graph state fields
- The LLM must produce a **clear, grounded, natural-language response** — it must not introduce facts not present in the tool output or state.
- Synthesis prompts and their outputs must be logged for observability.

### MCP Tools (`app/mcp_server/tools/`)

- Tools are organised into **category subdirectories** (`file_ops/`, `summarization/`, `context_retrieval/`). Each tool is its own package with three files: `tool.py` (handler), `schemas.py` (Pydantic I/O models), `__init__.py` (exports `get_tool() -> ToolSpec`).
- Every tool package must export `get_tool() -> ToolSpec`. The `ToolSpec` dataclass declares the tool's name, category, description, input schema class, handler callable, and optional `dependencies` dict (maps parameter names to other tool names resolved at dispatch time).
- Tools must declare **explicit Pydantic input and output schemas** in `schemas.py`.
- Tools must be **deterministic for the same inputs** unless explicitly documented as non-deterministic (e.g., live web search).
- Tools may call external APIs only when enabled by config.
- No shared mutable state between tools.
- To add a new tool: create a new package under the appropriate category, implement `get_tool()`, and restart the server (or call `POST /api/tools/refresh`). No changes to `server.py` or `tool_registry.py` are needed — discovery is automatic.

### LangGraph Nodes (`app/graph/nodes/`)

- Each node is produced by a **`make_node(dep)` factory** that closes over an injected dependency and returns the async callable `async def node(state: GraphState) -> dict`.
- Nodes must not import from `app/mcp_server/` or `app/llm/` directly — use injected client handles.
- Nodes return **partial state updates only** — never mutate the incoming state object.
- Read `state.get("model")` when passing an LLM call — this carries the per-request model override from the UI.
- Log at node entry and exit.

### Services

- DB and MQ clients are injected, not instantiated inside business logic.
- SQLite queries go through `sqlite_client.py` — no raw SQL in nodes or tools.
- MQ publish/consume operations go through `producer.py` and `consumer.py`.

---

## Operational Assumptions

- The system runs **fully locally by default** — no cloud account or remote service is required.
- **Ollama must be running** as a local service before the LangGraph workflow starts. Pull at least one model with `ollama pull <name>` before starting.
- **`OLLAMA_MODEL`** sets the server-wide default model used when a request carries no `model` field. The web UI dropdown lets users override this per-request without restarting the server; the list of available models is fetched live from Ollama via `GET /api/models`.
- **FastMCP server** runs as a separate long-lived process, independent of the LangGraph executor.
- **MCP tool dispatch** is handled by `MCPExecutor` (`app/services/mcp_executor.py`), which resolves tools from `ToolRegistry` and calls handlers via `inspect.signature`-based injection (injects `sandbox_root`, `llm`, `cache_client` and resolves `ToolSpec.dependencies` at dispatch time). The FastMCP server (`app/mcp_server/server.py`) exists as a separately deployable process if protocol-level isolation is required.
- **MCP tools may call external internet APIs** when the tool is configured and network access is available.
- **SQLite** is the default and only required database — no separate DB server needed.
- **Message queue is optional** — the system functions without it; MQ is added when async fan-out or background workers are needed.
- **Docker Compose** is the standard local runtime — `docker compose up` brings all services online.
- No cloud dependencies are assumed unless explicitly declared in `config.py` and documented in README.

---

## How to Extend

### Add a New MCP Tool

1. Choose or create a category directory under `app/mcp_server/tools/` (e.g. `file_ops/`, `summarization/`, `context_retrieval/`).
2. Create a new package directory: `app/mcp_server/tools/<category>/<tool_name>/`.
3. Add three files:
   - `schemas.py` — Pydantic input and output models.
   - `tool.py` — `async def run(...)` handler; declare only the parameters you need (`sandbox_root`, `llm`, `cache_client` are injected automatically by `MCPExecutor` when present in the signature).
   - `__init__.py` — exports `get_tool() -> ToolSpec` pointing at your schema and handler.
4. If the tool depends on another tool's handler (e.g. `refresh_cache` needs `fetch_web_resource`), declare it in `ToolSpec.dependencies = {"param_name": "other_tool_name"}`.
5. Restart the server **or** call `POST /api/tools/refresh` — `discover_all_tools()` picks up the new package automatically. No changes to `server.py` or `tool_registry.py` are needed.
6. Write a unit test in `tests/unit/test_tools.py`.

### Add a New LangGraph Node

1. Create a new file in `app/graph/nodes/`.
2. Implement the node as a pure function: `def my_node(state: GraphState) -> dict`.
3. Wire the node into `graph.py` with appropriate edges.
4. Add edge conditions to `app/graph/edges/` if the node introduces branching.
5. Write a unit test that passes a mock state and asserts the returned state delta.

### Add a New LLM Behavior

1. Add or modify prompt templates in `app/llm/`.
2. Update `tool_selector.py` for new tool-routing logic or `response_synthesizer.py` for new output patterns.
3. Accept a `model: str | None = None` parameter in any new LLM method and forward it to `OllamaClient.chat(model=model)` so the per-request model selection propagates correctly.
4. Write unit tests that assert correct behavior given mock LLM responses — do not depend on live Ollama calls in unit tests.

### Add a New Database Table

1. Add a migration script to `app/services/db/migrations/` (named sequentially, e.g. `002_add_jobs_table.sql`).
2. Update `sqlite_client.py` with any new query helpers.
3. Add a Pydantic model representing the row shape.
4. Write an integration test against an in-memory SQLite instance.

### Add a New Message Queue Consumer

1. Create a handler function in `app/services/mq/consumer.py`.
2. Register the handler for the relevant queue or topic.
3. Ensure idempotency — the handler must be safe to call more than once for the same message.
4. Write an integration test using a local MQ instance or mock.

### Add a New Topic to Context Retrieval

1. Open `app/mcp_server/resources/topic_map.yaml`.
2. Add an entry under `topics:` with:
   - `keywords` — list of lowercase phrases matched against user input (case-insensitive)
   - `urls` — ordered list of resource URLs to fetch for this topic
   - `ttl_seconds` — cache lifetime (e.g. `21600` for 6 h, `3600` for 1 h)
3. Restart the server — `topic_config.py` uses `@lru_cache` so the new config is picked up on next cold start. For a live reload, restart the FastAPI process.
4. Verify detection by sending a prompt containing one of the new keywords and checking the logs for `node_detect_topic_match`.

### Add a New External Integration

1. Implement the integration as an MCP tool in `app/mcp_server/tools/`.
2. Gate the integration behind a config flag in `app/utils/config.py`.
3. Document the required environment variables in README and `.env.example`.
4. Ensure the system degrades gracefully when the integration is disabled or unreachable.

---

## Architecture Trade-offs

### Two-Pass LLM Pattern (Fully LLM-Driven)

| | |
|---|---|
| **Pros** | Flexible; LLM handles novel inputs; no hard-coded routing rules |
| **Cons** | Two LLM calls per request increases latency; risk of tool hallucination; harder to debug |

### Hybrid Pattern (Recommended)

| | |
|---|---|
| **Pros** | Reliable tool selection; single LLM call for synthesis; fast and interpretable |
| **Cons** | Requires intent classification categories to be maintained as tools grow |

The hybrid pattern is the default approach in this project. The fully LLM-driven two-pass pattern is supported but should be opt-in per workflow, documented explicitly, and used only where dynamic tool selection genuinely adds value over rule-based routing.

---

## Quick Reference

| Concern | Location |
|---|---|
| MCP tools (file-system) | `app/mcp_server/tools/file_ops/` (list_files, read_file, search_files, extract_metadata) |
| MCP tools (summarisation) | `app/mcp_server/tools/summarization/` (summarize_file, summarize_text) |
| MCP tools (context retrieval) | `app/mcp_server/tools/context_retrieval/` (get_topic_resources, fetch_web_resource, get_cached_resource, refresh_cache) |
| Tool contract dataclass | `app/mcp_server/tool_spec.py` |
| Dynamic tool discovery | `app/mcp_server/tool_loader.py` |
| MCP server entry | `app/mcp_server/server.py` |
| MCP tool dispatcher | `app/services/mcp_executor.py` |
| Topic-to-URL config | `app/mcp_server/resources/topic_map.yaml` |
| Topic config loader + validator | `app/utils/topic_config.py` |
| Resource cache interface | `app/services/cache/cache_client.py` |
| Resource cache (SQLite) | `app/services/cache/sqlite_cache.py` |
| Ollama client | `app/llm/ollama_client.py` |
| Tool selection logic | `app/llm/tool_selector.py` |
| Response synthesis | `app/llm/response_synthesizer.py` |
| Live tool registry (LLM ↔ MCP bridge) | `app/llm/tool_registry.py` |
| Graph nodes | `app/graph/nodes/` |
| Graph state schema | `app/graph/state.py` |
| Graph wiring | `app/graph/graph.py` |
| API + model selection + tool refresh | `app/ui/api.py` |
| Web UI (model dropdown, tool panel) | `app/ui/static/index.html` |
| Database helpers | `app/services/db/sqlite_client.py` |
| MQ publish | `app/services/mq/producer.py` |
| MQ consume | `app/services/mq/consumer.py` |
| Config/env | `app/utils/config.py` |
| Logging setup | `app/utils/logging.py` |
| Error types | `app/utils/errors.py` |
