"""Graph node that runs first on every turn: blocks injection, redacts PII.

It rewrites the customer's message in state (same message id, so the reducer replaces
it), so the redacted text, not the raw text, is what the agents and the checkpoint see.
Sets `guard_blocked` / `guard_notice` fresh each turn; the graph ends early when blocked.
"""
from langchain_core.messages import AIMessage, HumanMessage

from ecommerce_agents import guards, identity
from ecommerce_agents.state import AgentState


def input_guard_node(state: AgentState, config) -> dict:
    last = state["messages"][-1] if state["messages"] else None
    clean = {"guard_blocked": False, "guard_notice": None}
    if not isinstance(last, HumanMessage) or not isinstance(last.content, str):
        return clean

    reason = guards.detect_injection(last.content)
    if reason:
        return {"guard_blocked": True, "guard_notice": None,
                "messages": [HumanMessage(content=guards.REMOVED, id=last.id),
                             AIMessage(content=guards.REFUSAL, name="input_guard")]}

    customer = identity.get_customer(identity.current_customer_id(config))
    pii = guards.redact_pii(last.content, own_email=customer["email"] if customer else None)
    if not pii.found:
        return clean
    return {"guard_blocked": False, "guard_notice": guards.pii_notice(pii.found),
            "messages": [HumanMessage(content=pii.text, id=last.id)]}


def route_after_guard(state: AgentState) -> str:
    return "__end__" if state.get("guard_blocked") else "supervisor"
