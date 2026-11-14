# Multi-Agent E-commerce Support System: Running Design Draft

Status: DISCUSSION. No code yet. Updated as decisions are made.

## 1. Goal (learning-first)
Build a multi-agent system step by step to learn and be able to explain in interviews:
supervisor architecture, agent communication, MCP tool calling, short/long-term memory,
failure handling, loop prevention, ReAct, human-in-the-loop, monitoring, guardrails,
agent permissions, evaluation, and Docker deployment (Render / Vercel).

## 2. Business problem
E-commerce customer support. One chat interface; a supervisor routes each message
to a specialist agent.

## 3. Agents
| Agent | Job | Access |
|---|---|---|
| Supervisor | Classify intent, route, merge multi-part answers, handle unclear requests | No business tools; routes only |
| Order-Status | Order status, tracking, delivery estimate | Read-only orders |
| Product-Info | Specs, price, stock, compatibility | Read-only catalog (maybe RAG) |
| Returns/Refund | Check return eligibility vs policy, initiate refund | Read orders + policy; write refunds |

## 4. Concept -> where it shows up
| Concept | Where |
|---|---|
| Supervisor pattern / communication | One LangGraph graph, shared state, supervisor picks next node |
| MCP | Orders DB and catalog exposed as MCP servers |
| ReAct loop | Refund agent: lookup -> policy check -> decide -> act, with iteration cap |
| Human-in-the-loop | Refunds above threshold pause graph (LangGraph interrupt) for approval |
| Permissions | Per-agent tool allowlists; only Refund agent has write tools |
| Guardrails | Prompt-injection check, PII redaction, output/policy validation |
| Failure handling | Retries/backoff, timeouts, 429 handling, fallbacks, escalate to human |
| Loop prevention | Max iterations, graph recursion limit, repeated-call detection |
| Short-term memory | Conversation state + checkpointer |
| Long-term memory | FUTURE: customer preferences across sessions |
| Monitoring | Tracing (e.g. Langfuse), latency, cost, errors |
| Evals | Routing accuracy, task success, tool-call correctness, policy compliance |
| Deployment | Docker; Render (backend), Vercel (UI) |

## 5. Tech stack (proposed)
- Python, LangGraph, FastAPI, MCP
- LLM: Groq via `langchain-groq` (decided)
- Data: SQLite with seeded mock data + short return-policy doc
- Later: simple UI (Streamlit or Next.js), Docker

## 6. Decision log
| # | Decision | Why | Alternatives |
|---|---|---|---|
| 1 | Business problem: e-commerce support | Familiar domain, clear read/write split, natural HITL (refunds) | IT helpdesk, invoice processing |
| 2 | Supervisor pattern | One orchestrator + domain specialists; clear routing and permission boundaries | Peer-to-peer / swarm, single agent with many tools |
| 4 | UI: Streamlit; refund threshold $50 | Fast to build in a 5-day window; AI-role focus | Next.js |
| 3 | LLM provider: Groq | Free tier, fast inference, suits many LLM calls (ReAct, evals) | Claude API, OpenAI, Ollama |

Groq notes: rate limits force us to build retry/backoff; open models have weaker
tool calling, which gives real failure/eval material; use a different (stronger)
model as eval judge; smaller model could handle routing/guardrail checks.
Verify current Groq model list at setup time.

## 7. Answers so far
- Comfortable with Python; Docker/async comfort unknown (assume beginner on Docker, explain it).
- Target: AI roles -> emphasize agents, evals, guardrails, observability over UI polish.
- Timeline: 5 days total.
- UI: Streamlit (decided; also keeps Vercel out, deploy all on Render).
- Refund HITL threshold: $50.

## 8. Roadmap (5 days, AI-role emphasis)
Day 1: Skeleton + mock SQLite data + LangGraph supervisor + Order-Status and Product-Info agents (Groq). Short-term memory via checkpointer.
Day 2: MCP servers for orders/catalog; Returns/Refund agent with ReAct loop; per-agent tool permissions; loop limits.
Day 3: Human-in-the-loop refund approval (> $50, LangGraph interrupt); guardrails (injection, PII, output checks); failure handling (retries, 429, timeouts, escalation).
Day 4: Monitoring/tracing (Langfuse); eval set + metrics (routing accuracy, task success, tool-call correctness, policy compliance); Streamlit UI.
Day 5: Dockerize, deploy to Render, README/architecture diagram, interview Q&A notes from the decision log. Long-term memory stays future scope.

Risk: 5 days is tight; if we slip, cut Streamlit polish and long-term memory first, never evals.
