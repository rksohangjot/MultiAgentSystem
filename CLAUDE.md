# CLAUDE.md

## Workflow rules
- Commit and push right after each completed step.
- One feature per session; start a new session for the next feature.
- Describe the approach in detail and get approval before implementing.
- Update the Status section below at the end of each feature.
- After each step: explain it in detail, name the file(s) (with line refs) holding the code, commit + push, then move to the next step.

## Project
Multi-agent e-commerce support system (learning-first, aimed at AI roles). One chat interface; a supervisor routes each message to a specialist agent. Python, LangGraph, Groq (`openai/gpt-oss-120b`), managed with `uv`. See `DESIGN.md` for goals, decision log and the 5-day roadmap.

## Commands
- Chat (CLI): `uv run python scripts/chat.py`
- Tests: `uv run pytest` (tests marked `llm` call the real Groq API and are skipped without `GROQ_API_KEY`)
- Config: copy `.env.example` to `.env` and set `GROQ_API_KEY` (and optionally `GROQ_MODEL`)

## Status: Day 2 (Refund agent) built

### Layout (`src/ecommerce_agents/`)
- `db.py`: SQLite with seeded mock data (5 customers, 10 products, 10 orders in various states). `ensure_db()` creates the DB on first use; `seed()` recreates it.
- `state.py`: shared `AgentState` (messages, `next_agent`, `iterations`). Agents communicate only through this.
- `llm.py`: Groq model factory, configured via `.env`.
- `supervisor.py`: structured-output routing to `order_status`, `product_info`, `refund` or `FINISH`. Answers greetings/off-topic directly. Loop prevention: capped at `MAX_ITERATIONS = 4`, then escalates to a human.
- `agents/order_status.py`: ReAct agent with read-only tool `get_order_status`.
- `agents/product_info.py`: ReAct agent with read-only tool `search_products`.
- `agents/refund.py`: ReAct refund agent (tools: `get_order_status`, `get_return_policy`, `check_return_eligibility`, `initiate_refund`); capped tool loop, hands off to a human when a refund needs approval.
- `tools/`: DB tools; per-agent allowlists (`AGENT_TOOLS`) ensure only the refund agent holds `initiate_refund`. Eligibility is deterministic code; refunds >$50 refused (stopgap).
- `graph.py`: supervisor -> specialists, each hands control back to the supervisor. In-memory checkpointer (short-term memory).
- `scripts/chat.py`: CLI chat loop that prints supervisor routing decisions.

### Tests (`tests/`)
- `test_tools.py`: order found/missing, product search, no match (temp seeded DB).
- `test_graph.py`: iteration cap, direct greeting reply, route mapping (stubbed router), plus live routing tests (order, product, 2 refund) and a live mixed-message test (order_status + refund).
- `test_refund_graph.py` and refund tool/permission tests (stubbed LLM): eligibility, idempotency, $50 cap, allowlists, mixed request order_status -> refund.
- Last run: 37 passed, 8 skipped (live Groq tests need GROQ_API_KEY; all 8 live tests pass on gpt-oss-120b).

### Not built yet
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
- Decision: until human-in-the-loop approval exists, `initiate_refund` must refuse refunds over $50 with a "needs human approval" message (stopgap). Refund agent is built; next feature: human-in-the-loop approval (replaces this stopgap).

## Approved plan: Returns/Refund agent (done; kept for reference)
Design decisions:
- Eligibility is decided by deterministic code, not the LLM. `check_return_eligibility(order_id)` -> `{eligible, reason, amount}`: order must be `delivered`, within 30 days of delivery, no existing refund.
- `initiate_refund(order_id, reason)` is the only write tool. It re-runs the eligibility check itself, refuses amounts over $50 ("needs human approval"), and is idempotent (a repeat call returns the existing refund).
- "Today" is injectable (default real date) so tests are deterministic.
- Full refunds only. No partial refunds or multi-item orders.
- Refund agent tools: `get_order_status`, `get_return_policy`, `check_return_eligibility`, `initiate_refund`. No product lookup.
- Identity check deferred: the order ID alone is accepted for now (known gap; revisit with guardrails/permissions).
- Add a per-agent tool allowlist with a test that only the refund agent holds `initiate_refund`.
- Cap the refund agent's tool calls/recursion in addition to the supervisor's 4-turn cap.

Steps (commit + push after each):
1. `refunds` table, `data/return_policy.md`, seed cases (incl. outside the window and already refunded).
2. Eligibility + refund tools with unit tests.
3. Refund agent and prompt.
4. Graph wiring, supervisor routing, permission test.
5. Live routing test, then update this Status section.
Also test a mixed message ("order is late, I want my money back") routes to both agents.
