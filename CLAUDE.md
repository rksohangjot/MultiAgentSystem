# CLAUDE.md

## Workflow rules
- Commit and push right after each completed step.
- One feature per session; start a new session for the next feature.
- Describe the approach in detail and get approval before implementing.
- Update the Status section below at the end of each feature.

## Project
Multi-agent e-commerce support system (learning-first, aimed at AI roles). One chat interface; a supervisor routes each message to a specialist agent. Python, LangGraph, Groq (`llama-3.3-70b-versatile`), managed with `uv`. See `DESIGN.md` for goals, decision log and the 5-day roadmap.

## Commands
- Chat (CLI): `uv run python scripts/chat.py`
- Tests: `uv run pytest` (tests marked `llm` call the real Groq API and are skipped without `GROQ_API_KEY`)
- Config: copy `.env.example` to `.env` and set `GROQ_API_KEY` (and optionally `GROQ_MODEL`)

## Status: Day 1 of roadmap built

### Layout (`src/ecommerce_agents/`)
- `db.py`: SQLite with seeded mock data (5 customers, 10 products, 10 orders in various states). `ensure_db()` creates the DB on first use; `seed()` recreates it.
- `state.py`: shared `AgentState` (messages, `next_agent`, `iterations`). Agents communicate only through this.
- `llm.py`: Groq model factory, configured via `.env`.
- `supervisor.py`: structured-output routing to `order_status`, `product_info` or `FINISH`. Answers greetings/off-topic directly. Loop prevention: capped at `MAX_ITERATIONS = 4`, then escalates to a human.
- `agents/order_status.py`: ReAct agent with read-only tool `get_order_status`.
- `agents/product_info.py`: ReAct agent with read-only tool `search_products`.
- `tools/`: the read-only DB tools used by the agents.
- `graph.py`: supervisor -> specialists, each hands control back to the supervisor. In-memory checkpointer (short-term memory).
- `scripts/chat.py`: CLI chat loop that prints supervisor routing decisions.

### Tests (`tests/`)
- `test_tools.py`: order found/missing, product search, no match (temp seeded DB).
- `test_graph.py`: iteration cap, direct greeting reply, route mapping (stubbed router), plus 2 live routing tests.
- Last run: 7 passed, 2 skipped (live Groq tests need GROQ_API_KEY).

### Not built yet
- Returns/Refund agent with ReAct loop
- MCP servers for orders/catalog
- Human-in-the-loop refund approval (> $50, LangGraph interrupt)
- Guardrails (injection, PII, output checks)
- Failure handling (retries/backoff, 429, timeouts, escalation)
- Tracing (Langfuse) and evals
- Streamlit UI
- Docker / Render deployment
- Long-term memory (future scope)

## Notes
- Git repo on branch `main`, remote `origin` = https://github.com/rksohangjot/MultiAgentSystem.git
- Design constraints: only the Refund agent should have write tools; use a stronger model as eval judge; never cut evals if the schedule slips.
