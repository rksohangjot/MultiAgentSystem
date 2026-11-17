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

## Status: Day 3 built (Guardrails + failure handling)

### Layout (`src/ecommerce_agents/`)
- `db.py`: SQLite with seeded mock data (5 customers, 10 products, 10 orders in various states). `ensure_db()` creates the DB on first use; `seed()` recreates it.
- `identity.py`: customer identity. `owns_order(order_id, customer_id)`, `find_customer_by_email`. `customer_id` comes from the run config (`config["configurable"]["customer_id"]`), never from the LLM; tools fail closed without it, and a non-owned order returns the same "not found" as a missing one.
- `guards.py`: pure input guard. `detect_injection` (regex heuristics, no LLM), `redact_pii` (Luhn-checked cards, SSN, phone, other people's emails; own email kept).
- `input_guard.py`: first graph node. Injection -> message replaced, refusal, turn ends (supervisor never runs). PII -> message rewritten redacted + `guard_notice` shown to the customer. `new_message()` also redacts, because LangGraph checkpoints the raw input before any node runs.
- `output_guard.py`: checks every reply before the customer sees it (deterministic). Blocks unbacked refund claims (says refunded/approved but no `completed` refund for that customer's order; hedged/negated sentences are ignored), internal leaks (tool names, prompt text, state fields), other customers' emails/names, raw card/SSN/phone. A blocked reply is replaced (same id) by `FALLBACK` (escalates to a human). `guarded(node)` wraps supervisor, order_status, product_info and refund nodes in `graph.py`; the approval node is code-generated from the DB so it is not wrapped.
- `state.py`: shared `AgentState` (messages, `next_agent`, `iterations`, `escalated`, `guard_blocked`, `guard_notice`, `pending_refund_id`). Agents communicate only through this.
- `llm.py`: Groq model factory (`.env`). Explicit `max_retries=2` (3 attempts) and `request_timeout=20s`, overridable via `GROQ_MAX_RETRIES` / `GROQ_TIMEOUT_SECONDS`. The Groq SDK does the retrying (408/409/429/5xx/connection, backoff, Retry-After; never 400/401/403). Tools are never retried.
- `errors.py`: `is_api_error`, `is_transient`, `reason` for Groq API errors.
- `resilience.py`: `escalate_on_api_error(node, name)`. A Groq API error that survives the retries becomes one fixed handoff message (`HANDOFF`), `escalated=True`, `next_agent=FINISH`, and the turn ends (no second call to the unavailable model). Transient -> WARNING log, permanent (e.g. bad key) -> ERROR log, same customer text. Bugs and LangGraph interrupts are not caught. If the refund run dies after `initiate_refund` queued a >$50 refund (model outage or loop cap), `orphaned_refund_update` (agents/refund.py) re-finds the `pending_approval` row via `pending_refund_for_customer` and routes it to the approval node anyway; after approval an escalated turn ends instead of asking the supervisor again. Wraps supervisor + the three specialists inside the output guard.
- `supervisor.py`: structured-output routing to `order_status`, `product_info`, `refund` or `FINISH`. Answers greetings/off-topic directly. Loop prevention: capped at `MAX_ITERATIONS = 4`, then escalates to a human.
- `agents/order_status.py`: ReAct agent with read-only tool `get_order_status`.
- `agents/product_info.py`: ReAct agent with read-only tool `search_products`.
- `agents/refund.py`: ReAct refund agent (tools: `get_order_status`, `get_return_policy`, `check_return_eligibility`, `initiate_refund`); capped tool loop, hands off to a human when the loop cap is hit. Sets `pending_refund_id` when `initiate_refund` queues a refund.
- `tools/`: DB tools; per-agent allowlists (`AGENT_TOOLS`) ensure only the refund agent holds `initiate_refund`. Eligibility is deterministic code; refunds >$50 are saved as `pending_approval` (not paid) and decided by a human via `decide_refund` (plain code, NOT an agent tool; idempotent; re-checks the window; rejection is final).
- `approval.py`: `approval_node` pauses with LangGraph `interrupt()` for a reviewer decision `{approved, note}`, applies it via `decide_refund`, tells the customer. Everything before `interrupt()` only reads (a resumed node re-runs from the top).
- `graph.py`: supervisor -> specialists, each hands control back to the supervisor; `refund` -> `approval` when a refund is pending. `build_graph()` defaults to in-memory checkpointer; `sqlite_checkpointer(path)` is durable (a pause survives restart).
- `chat_loop.py`: CLI logic (`stream_turn`, `cli_review`, `resume_pending`, `sign_in` email stand-in login, `session_config` per-customer thread) so it is testable.
- `scripts/chat.py`: thin CLI over `chat_loop.py`; prints routing, you play the human reviewer for refunds >$50; state in `data/checkpoints.db`.

### Tests (`tests/`)
- `test_tools.py`: order found/missing, product search, no match (temp seeded DB).
- `test_graph.py`: iteration cap, direct greeting reply, route mapping (stubbed router), plus live routing tests (order, product, 2 refund) and a live mixed-message test (order_status + refund).
- `test_refund_graph.py` and refund tool/permission tests (stubbed LLM): eligibility, idempotency, $50 cap, allowlists, mixed request order_status -> refund.
- `test_failures.py`: SDK retry semantics against a mock transport (429->ok, 3 attempts, 5xx retried, 400/401/403 not), error classification, graph handoff for supervisor/order/refund failures, flag reset, bugs not swallowed.
- `test_output_guard.py`: claim vs DB (pending/rejected don't back a paid claim), hedge table, leaks, other-customer data, graph replacement, honest flow not blocked.
- `test_input_guard.py`: injection attack/benign tables, PII cases, graph wiring (blocked before supervisor, redacted before supervisor and in checkpoints, flags reset).
- `test_identity.py`: ownership checks, fail-closed, LLM can't set customer_id, sign-in.
- `test_approval.py`, HITL section of `test_refund_graph.py`, `test_chat_loop.py`: pause/approve/reject, restart resume, reviewer prompt, plus a live over-$50 test.
- Last run: 161 passed incl. 9 live tests (need GROQ_API_KEY) pass on gpt-oss-120b.

### Not built yet
- MCP servers for orders/catalog
- Real reviewer UI / auth for reviewers (CLI user is the reviewer for now; arrives with Streamlit)
- Guardrails: all three layers DONE (identity, input guard, output guard). Known limits: heuristics are regex, so paraphrased injections and oddly worded refund claims can slip past; tool permissions and code-enforced policy are the backstop. Blocked replies are only logged (`logging`), not traced yet.
- Failure handling gaps: (1) timeout is per request (20s x 3 attempts), not per turn: a per-turn deadline would wrongly expire during a human-approval pause; (2) Groq 400 `tool_use_failed` (malformed tool call) is treated as permanent, not retried.
- Tracing (Langfuse) and evals
- Streamlit UI
- Docker / Render deployment
- Long-term memory (future scope)

## Notes
- Git repo on branch `main`, remote `origin` = https://github.com/rksohangjot/MultiAgentSystem.git
- Design constraints: only the Refund agent should have write tools; use a stronger model as eval judge; never cut evals if the schedule slips.
- Decision: until human-in-the-loop approval exists, `initiate_refund` must refuse refunds over $50 with a "needs human approval" message (stopgap). Superseded: HITL approval is built (`approval.py`); >$50 refunds are queued and decided by a human. Guardrails are built. Failure handling is built. Next feature: tracing (Langfuse) and evals.

## Approved plan: Returns/Refund agent (done; kept for reference)
Design decisions:
- Eligibility is decided by deterministic code, not the LLM. `check_return_eligibility(order_id)` -> `{eligible, reason, amount}`: order must be `delivered`, within 30 days of delivery, no existing refund.
- `initiate_refund(order_id, reason)` is the only write tool. It re-runs the eligibility check itself, queues amounts over $50 for human approval, and is idempotent (a repeat call returns the existing refund).
- "Today" is injectable (default real date) so tests are deterministic.
- Full refunds only. No partial refunds or multi-item orders.
- Refund agent tools: `get_order_status`, `get_return_policy`, `check_return_eligibility`, `initiate_refund`. No product lookup.
- Identity check: done (guardrails step 1); email lookup is a stand-in for real auth.
- Add a per-agent tool allowlist with a test that only the refund agent holds `initiate_refund`.
- Cap the refund agent's tool calls/recursion in addition to the supervisor's 4-turn cap.

Steps (commit + push after each):
1. `refunds` table, `data/return_policy.md`, seed cases (incl. outside the window and already refunded).
2. Eligibility + refund tools with unit tests.
3. Refund agent and prompt.
4. Graph wiring, supervisor routing, permission test.
5. Live routing test, then update this Status section.
Also test a mixed message ("order is late, I want my money back") routes to both agents.
