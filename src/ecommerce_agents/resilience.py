"""When a model call still fails after the SDK's retries, hand the customer to a human.

`escalate_on_api_error(node)` catches Groq API errors only (programming bugs still raise,
and LangGraph's interrupt/recursion signals pass through). The failure becomes one fixed
customer message, `escalated=True`, and `next_agent="FINISH"`, so the graph ends the turn
instead of asking the same unavailable model again.
Transient and permanent errors get the same customer message; the log tells them apart
(a permanent one, e.g. a bad API key, is logged at ERROR because it is a config bug).
"""
import logging

from langchain_core.messages import AIMessage

from ecommerce_agents import errors

log = logging.getLogger(__name__)

HANDOFF = "Sorry, I'm having technical trouble right now. Let me get a human to help."


def escalate_on_api_error(node, name: str):
    """Wrap a `(state, config) -> update` node. `name` labels the node in logs and messages."""
    def wrapper(state, config):
        try:
            return node(state, config)
        except Exception as exc:  # noqa: BLE001 - narrowed immediately below
            if not errors.is_api_error(exc):
                raise
            level = logging.WARNING if errors.is_transient(exc) else logging.ERROR
            log.log(level, "node %s failed (%s); escalating to a human", name, errors.reason(exc))
            return {"messages": [AIMessage(content=HANDOFF, name=name)],
                    "escalated": True, "next_agent": "FINISH"}
    return wrapper
