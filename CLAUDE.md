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

### High-Level Workflow (Two-Pass LLM Pattern)

The core execution pattern uses the local Ollama LLM in two distinct passes per request:

```
User Input
    │
    ▼
[LangGraph Node: Tool Selection]
    │  Ollama LLM receives user input + available tool schemas
    │  → selects tool + generates arguments
    ▼
[LangGraph Node: MCP Tool Invocation]
    │  MCP client calls selected tool
    │  → returns structured output
    ▼
[LangGraph Node: Response Synthesis]
    │  Ollama LLM receives: user input + tool output + graph state
    │  → generates natural-language response
    ▼
Final Response
```

**Pass 1 — Tool Selection:** The LLM analyzes user input against available MCP tool schemas and returns a structured tool call (name + arguments). Output is validated against the tool's Pydantic schema before execution.

**Pass 2 — Response Synthesis:** The LLM receives the original user input plus the MCP tool's structured output as enriched context and produces the final natural-language response.

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
            /tools              # Individual MCP tool definitions
            /resources          # MCP resource handlers
            /prompts            # MCP prompt templates
            server.py           # FastMCP server entry point
        /graph
            /nodes              # LangGraph node functions (pure, testable)
            /edges              # Conditional edge logic
            state.py            # Shared graph state schema (Pydantic)
            graph.py            # Graph assembly and compilation
        /llm
            ollama_client.py        # Wrapper for local Ollama LLM calls
            tool_selector.py        # LLM-driven tool selection logic
            response_synthesizer.py # LLM natural-language output generator
        /services
            /db
                sqlite_client.py    # SQLite connection and query helpers
                /migrations         # Schema migration scripts
            /mq
                producer.py         # Message queue publish interface
                consumer.py         # Message queue subscribe/consume logic
        /utils
            logging.py          # Structured logging setup
            config.py           # Environment and settings loader
            errors.py           # Shared exception types
    /tests
        /unit                   # Tests for nodes, tools, LLM logic, services
        /integration            # End-to-end graph and MCP integration tests
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

- Each tool lives in its own file.
- Tools must declare **explicit Pydantic input and output schemas**.
- Tools must be **deterministic for the same inputs** unless explicitly documented as non-deterministic (e.g., live web search).
- Tools may call external APIs only when enabled by config.
- No shared mutable state between tools.

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
- **MCP tool dispatch** is handled by `MCPExecutor` (`app/services/mcp_executor.py`), which calls tool functions via direct Python import rather than the MCP client protocol. The FastMCP server (`app/mcp_server/server.py`) exists as a separately deployable process if protocol-level isolation is required.
- **MCP tools may call external internet APIs** when the tool is configured and network access is available.
- **SQLite** is the default and only required database — no separate DB server needed.
- **Message queue is optional** — the system functions without it; MQ is added when async fan-out or background workers are needed.
- **Docker Compose** is the standard local runtime — `docker compose up` brings all services online.
- No cloud dependencies are assumed unless explicitly declared in `config.py` and documented in README.

---

## How to Extend

### Add a New MCP Tool

1. Create a new file in `app/mcp_server/tools/`.
2. Define Pydantic input and output schemas.
3. Register the tool in `server.py` using the FastMCP decorator or registration API.
4. Add the tool schema to the tool-selection prompt in `tool_selector.py`.
5. Write a unit test in `tests/unit/`.

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
| MCP tools | `app/mcp_server/tools/` |
| MCP server entry | `app/mcp_server/server.py` |
| MCP tool dispatcher | `app/services/mcp_executor.py` |
| Ollama client | `app/llm/ollama_client.py` |
| Tool selection logic | `app/llm/tool_selector.py` |
| Response synthesis | `app/llm/response_synthesizer.py` |
| Tool registry (LLM ↔ MCP bridge) | `app/llm/tool_registry.py` |
| Graph nodes | `app/graph/nodes/` |
| Graph state schema | `app/graph/state.py` |
| Graph wiring | `app/graph/graph.py` |
| API + model selection endpoint | `app/ui/api.py` |
| Web UI + model dropdown | `app/ui/static/index.html` |
| Database helpers | `app/services/db/sqlite_client.py` |
| MQ publish | `app/services/mq/producer.py` |
| MQ consume | `app/services/mq/consumer.py` |
| Config/env | `app/utils/config.py` |
| Logging setup | `app/utils/logging.py` |
| Error types | `app/utils/errors.py` |
