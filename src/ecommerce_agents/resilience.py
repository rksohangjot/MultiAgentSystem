"""When a model call still fails after the SDK's retries, hand the customer to a human.

`escalate_on_api_error(node)` catches Groq API errors only (programming bugs still raise,
and LangGraph's interrupt/recursion signals pass through). The failure becomes one fixed
customer message, `escalated=True`, and `next_agent="FINISH"`, so the graph ends the turn
instead of asking the same unavailable model again.
A malformed tool call (400 `tool_use_failed`) is retried first, see below.
Transient and permanent errors get the same customer message; the log tells them apart
(a permanent one, e.g. a bad API key, is logged at ERROR because it is a config bug).
"""
import logging
import os

from langchain_core.messages import AIMessage

from ecommerce_agents import errors, tracing

log = logging.getLogger(__name__)

DEFAULT_TOOL_CALL_RETRIES = 2  # extra attempts after a Groq 400 tool_use_failed => 3 attempts per node run

HANDOFF = "Sorry, I'm having technical trouble right now. Let me get a human to help."


def escalate_on_api_error(node, name: str, extra=None):
    """Wrap a `(state, config) -> update` node. `name` labels the node in logs and messages.

    `extra(state, config) -> dict` adds node-specific state to the handoff update (the refund
    node uses it to keep a queued refund reachable by the approval node).

    A malformed tool call (Groq 400 `tool_use_failed`) is a sampling failure, so the whole node is
    re-run up to GROQ_TOOL_CALL_RETRIES more times before escalating. Re-running is safe: a raised
    node returns no update (state and the supervisor's iteration count are untouched), tools are
    never retried on their own, and `initiate_refund` is idempotent.
    """
    def wrapper(state, config):
        retries = int(os.getenv("GROQ_TOOL_CALL_RETRIES", DEFAULT_TOOL_CALL_RETRIES))
        attempt = 0
        while True:
            try:
                return node(state, config)
            except Exception as exc:  # noqa: BLE001 - narrowed immediately below
                if not errors.is_api_error(exc):
                    raise
                if errors.is_tool_call_failure(exc) and attempt < retries:
                    attempt += 1
                    log.warning("node %s: malformed tool call from the model; retry %d/%d",
                                name, attempt, retries)
                    tracing.record_event("retry.tool_call_failed", node=name, attempt=attempt)
                    continue
                level = logging.WARNING if errors.is_transient(exc) else logging.ERROR
                log.log(level, "node %s failed (%s); escalating to a human", name, errors.reason(exc))
                tracing.record_event("escalation.api_error", node=name, reason=errors.reason(exc),
                                     transient=errors.is_transient(exc))
                update = {"messages": [AIMessage(content=HANDOFF, name=name)],
                          "escalated": True, "next_agent": "FINISH"}
                return {**update, **extra(state, config)} if extra else update
    return wrapper
