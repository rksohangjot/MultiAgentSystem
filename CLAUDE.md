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

## Status: Day 3 built (Guardrails + failure handling) + tracing (Langfuse) + evals (harness, 35 cases, Qwen judge) + tool_use_failed retry

### Layout (`src/ecommerce_agents/`)
- `db.py`: SQLite with seeded mock data (5 customers, 10 products, 10 orders in various states). `ensure_db()` creates the DB on first use; `seed()` recreates it.
- `identity.py`: customer identity. `owns_order(order_id, customer_id)`, `find_customer_by_email`. `customer_id` comes from the run config (`config["configurable"]["customer_id"]`), never from the LLM; tools fail closed without it, and a non-owned order returns the same "not found" as a missing one.
- `guards.py`: pure input guard. `detect_injection` (regex heuristics, no LLM), `redact_pii` (Luhn-checked cards, SSN, phone, other people's emails; own email kept).
- `input_guard.py`: first graph node. Injection -> message replaced, refusal, turn ends (supervisor never runs). PII -> message rewritten redacted + `guard_notice` shown to the customer. `new_message()` also redacts, because LangGraph checkpoints the raw input before any node runs.
- `output_guard.py`: checks every reply before the customer sees it (deterministic). Blocks unbacked refund claims (says refunded/approved but no `completed` refund for that customer's order; hedged/negated sentences are ignored), internal leaks (tool names, prompt text, state fields), other customers' emails/names, raw card/SSN/phone. A blocked reply is replaced (same id) by `FALLBACK` (escalates to a human). `guarded(node)` wraps supervisor, order_status, product_info and refund nodes in `graph.py`; the approval node is code-generated from the DB so it is not wrapped.
- `state.py`: shared `AgentState` (messages, `next_agent`, `iterations`, `escalated`, `guard_blocked`, `guard_notice`, `pending_refund_id`). Agents communicate only through this.
- `llm.py`: Groq model factory (`.env`). Explicit `max_retries=2` (3 attempts) and `request_timeout=20s`, overridable via `GROQ_MAX_RETRIES` / `GROQ_TIMEOUT_SECONDS`. The Groq SDK does the retrying (408/409/429/5xx/connection, backoff, Retry-After; never 400/401/403). Tools are never retried.
- `errors.py`: `is_api_error`, `is_transient`, `reason` for Groq API errors.
- `resilience.py`: `escalate_on_api_error(node, name)`. A Groq API error that survives the retries becomes one fixed handoff message (`HANDOFF`), `escalated=True`, `next_agent=FINISH`, and the turn ends (no second call to the unavailable model). Transient -> WARNING log, permanent (e.g. bad key) -> ERROR log, same customer text. Bugs and LangGraph interrupts are not caught. If the refund run dies after `initiate_refund` queued a >$50 refund (model outage or loop cap), `orphaned_refund_update` (agents/refund.py) re-finds the `pending_approval` row via `pending_refund_for_customer` and routes it to the approval node anyway; after approval an escalated turn ends instead of asking the supervisor again. Wraps supervisor + the three specialists inside the output guard. A Groq 400 `tool_use_failed` (`errors.is_tool_call_failure`; malformed tool call, also hits the supervisor's structured output) is a sampling failure: the wrapper re-runs the node up to `GROQ_TOOL_CALL_RETRIES` (default 2) more times (WARNING log + `retry.tool_call_failed` event: node + attempt only) before the handoff above. Safe because a raised node returns no update (iterations not burned) and `initiate_refund` is idempotent; a re-run of a refund that was already queued still reaches approval. Other 400s are not retried.
- `tracing.py`: optional Langfuse tracing, off unless `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are set (`LANGFUSE_BASE_URL` for the host). `callbacks()` returns the LangChain handler (or `[]`; a setup failure is logged and tracing skipped, never breaks a turn); the client is created once with `mask` (recursive `redact_pii`) as a second PII layer. `session_config` (chat_loop.py) adds the handler plus `langfuse_session_id` (thread), `langfuse_user_id` (`customer-<id>`, no email/name) and tags. `record_event(name, **detail)` marks categories only (never text) on the current trace: `guard.input.injection_blocked`, `guard.input.pii_redacted`, `guard.output.blocked`, `escalation.api_error`, `escalation.loop_cap`. `scripts/chat.py` calls `flush()` on exit. Verified live: one trace per `graph.stream` call, an approval resume is a separate trace in the same session, events land on the right turn, no raw card in any observation.
- `supervisor.py`: structured-output routing to `order_status`, `product_info`, `refund` or `FINISH`. Answers greetings/off-topic directly. Loop prevention: capped at `MAX_ITERATIONS = 4`, then escalates to a human (`escalated=True`). A specialist runs at most once per customer message, enforced in code (`specialists_run_this_turn`, read from message names since the latest human message; a repeat becomes FINISH + `supervisor.repeat_blocked` event). Before this, the LLM re-routed a finished refund to the refund agent and its stale "queued for review" reply overwrote the approval outcome.
- `agents/order_status.py`: ReAct agent with read-only tool `get_order_status`.
- `agents/product_info.py`: ReAct agent with read-only tool `search_products`.
- `agents/refund.py`: ReAct refund agent (tools: `get_order_status`, `get_return_policy`, `check_return_eligibility`, `initiate_refund`); capped tool loop, hands off to a human when the loop cap is hit. Sets `pending_refund_id` when `initiate_refund` queues a refund.
- `tools/`: DB tools; per-agent allowlists (`AGENT_TOOLS`) ensure only the refund agent holds `initiate_refund`. Eligibility is deterministic code; refunds >$50 are saved as `pending_approval` (not paid) and decided by a human via `decide_refund` (plain code, NOT an agent tool; idempotent; re-checks the window; rejection is final).
- `approval.py`: `approval_node` pauses with LangGraph `interrupt()` for a reviewer decision `{approved, note}`, applies it via `decide_refund`, tells the customer. Everything before `interrupt()` only reads (a resumed node re-runs from the top).
- `graph.py`: supervisor -> specialists, each hands control back to the supervisor; `refund` -> `approval` when a refund is pending. `build_graph()` defaults to in-memory checkpointer; `sqlite_checkpointer(path)` is durable (a pause survives restart).
- `chat_loop.py`: CLI logic (`stream_turn`, `cli_review`, `resume_pending`, `sign_in` email stand-in login, `session_config` per-customer thread) so it is testable.
- `scripts/chat.py`: thin CLI over `chat_loop.py`; prints routing, you play the human reviewer for refunds >$50; state in `data/checkpoints.db`.

- `evals/` (run with `PYTHONPATH=src uv run python scripts/run_evals.py [--category X] [--case ID] [--no-judge] [--list]`; the package is not installed, so plain `uv run python scripts/...` needs `PYTHONPATH=src`; not part of pytest, calls the live API): `case.py` (`EvalCase`/`Observation`), `runner.py` (`run_case`: real graph on a throwaway seeded DB, pinned date, scripted reviewer decisions), `checks.py` (deterministic checkers driven by each case's `expect`; DB/tool/route/flag checks and `reply_must_contain` are strict, `reply_contains_any`/`reply_excludes` are warnings only), `dataset.py` (35 cases: routing 8, refund_policy 8, mixed 4, permissions 5, adversarial 10), `judge.py` (Qwen `qwen/qwen3.8-27b` via `JUDGE_MODEL`, faithfulness/helpfulness/tone 1-5 on the 10 `judge=True` cases, informational only, never changes pass/fail, a failed judge call is recorded not raised), `report.py` (summary by category and metric, JSON report to `data/eval_reports/`, git-ignored). Groq has no open model stronger than gpt-oss-120b, so the judge is a different family but smaller; calibration tests check it discriminates.

### Tests (`tests/`)
- `test_tools.py`: order found/missing, product search, no match (temp seeded DB).
- `test_graph.py`: iteration cap, direct greeting reply, route mapping (stubbed router), plus live routing tests (order, product, 2 refund) and a live mixed-message test (order_status + refund).
- `test_refund_graph.py` and refund tool/permission tests (stubbed LLM): eligibility, idempotency, $50 cap, allowlists, mixed request order_status -> refund.
- `test_failures.py`: SDK retry semantics against a mock transport (429->ok, 3 attempts, 5xx retried, 400/401/403 not), error classification, graph handoff for supervisor/order/refund failures, flag reset, bugs not swallowed.
- `test_output_guard.py`: claim vs DB (pending/rejected don't back a paid claim), hedge table, leaks, other-customer data, graph replacement, honest flow not blocked.
- `test_input_guard.py`: injection attack/benign tables, PII cases, graph wiring (blocked before supervisor, redacted before supervisor and in checkpoints, flags reset).
- `test_identity.py`: ownership checks, fail-closed, LLM can't set customer_id, sign-in.
- `test_approval.py`, HITL section of `test_refund_graph.py`, `test_chat_loop.py`: pause/approve/reject, restart resume, reviewer prompt, plus a live over-$50 test.
- `test_eval_checks.py`, `test_eval_dataset.py` (every case well-formed), `test_eval_report.py`, `test_eval_runner.py`, `test_eval_judge.py` (offline stub plus 2 live calibration tests), `test_supervisor_repeat.py` (once-per-turn rule, loop-cap escalation, approved-refund regression), `test_tool_call_retry.py` (real SDK exceptions via mock transport; retry, exhaustion, config, refund still reaches approval).
- `test_tracing.py`: on/off, setup failure, mask, session_config ids, event helpers swallow errors, guard/escalation events carry categories only (fake client, offline).
- Last run: 275 non-live tests pass (11 live tests deselected; the 2 judge calibration tests were run and passed). Last full eval run: 35/35 cases, judge averages faithfulness 5.0 / helpfulness 4.3 / tone 4.9 (one run, the model is not deterministic).

### Not built yet
- MCP servers for orders/catalog
- Real reviewer UI / auth for reviewers (CLI user is the reviewer for now; arrives with Streamlit)
- Guardrails: all three layers DONE (identity, input guard, output guard). Known limits: heuristics are regex, so paraphrased injections and oddly worded refund claims can slip past; tool permissions and code-enforced policy are the backstop. Blocked replies are logged and recorded as Langfuse events.
- Failure handling gaps: (1) timeout is per request (20s x 3 attempts), not per turn: a per-turn deadline would wrongly expire during a human-approval pause; (2) DONE: Groq 400 `tool_use_failed` is retried twice at node level (live rate is about 1 in 70 case runs, so it is covered by offline tests, not a before/after eval).
- Evals: DONE (see `evals/` above). Eval findings: the judge caught two bugs the deterministic checks missed (stale reply after an approved refund; loop-cap handoff not marked escalated), both fixed with regression tests and strict eval expectations. Open eval notes: `refund_over_50_rejected` helpfulness 2 (rejection message gives no next step); `mixed_late_and_refund` expects the refund agent alone because it holds `get_order_status` (the supervisor's extra order_status route varies run to run). Tracing gaps: the CLI redacts PII in `new_message()` before the graph, so `guard.input.pii_redacted` never fires on that path; refund loop-cap event has no test; Langfuse REST `/api/public/traces` is gone (410), read traces via `Langfuse().api.observations.get_many`.
- Streamlit UI
- Docker / Render deployment
- Long-term memory (future scope)

## Notes
- Git repo on branch `main`, remote `origin` = https://github.com/rksohangjot/MultiAgentSystem.git
- Design constraints: only the Refund agent should have write tools; use a stronger model as eval judge; never cut evals if the schedule slips.
- Decision: until human-in-the-loop approval exists, `initiate_refund` must refuse refunds over $50 with a "needs human approval" message (stopgap). Superseded: HITL approval is built (`approval.py`); >$50 refunds are queued and decided by a human. Guardrails are built. Failure handling is built. Tracing is built. Evals are built. The tool_use_failed retry is built. Next feature: Streamlit UI (reviewer panel replaces the CLI stand-in), then Docker/Render deployment, then MCP servers if time allows.

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
