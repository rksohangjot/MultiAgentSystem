"""Logic behind the web UI, kept free of Streamlit so it can be tested.

Two surfaces share one graph and one durable checkpointer:
  - the customer chat: a turn ends at the approval pause ("queued for review");
  - the reviewer queue: finds paused conversations and resumes the right customer's thread.
The customer id always comes from the signed-in session (server side), never from chat text.
"""
import logging
from typing import Iterator

from langgraph.types import Command

from ecommerce_agents import identity
from ecommerce_agents.chat_loop import pending_review, session_config
from ecommerce_agents.tools.refund_tools import pending_refunds

log = logging.getLogger(__name__)
SURFACES = ("web", "cli")  # where a paused conversation may live


def web_config(customer: dict) -> dict:
    return session_config(customer, surface="web")


def history(graph, cfg) -> list[dict]:
    """Customer-visible transcript of a thread: human and AI messages that have text (tool calls don't)."""
    values = graph.get_state(cfg).values or {}
    out = []
    for m in values.get("messages", []):
        if m.type in ("human", "ai") and m.content:
            out.append({"role": "user" if m.type == "human" else "assistant", "content": str(m.content)})
    return out


def has_pending(graph, cfg) -> bool:
    return pending_review(graph, cfg) is not None


def turn_events(graph, graph_input, cfg) -> Iterator[dict]:
    """Run one turn and yield UI events; the turn stops at an approval pause instead of asking anyone.

    Events: {"type": "notice", "text"} (e.g. personal data removed), {"type": "route", "agent"},
    {"type": "reply", "node", "text"}, and a final {"type": "pending", "payload"} if paused.
    """
    for step in graph.stream(graph_input, cfg, stream_mode="updates"):
        for node, upd in step.items():
            upd = upd or {}
            if node == "__interrupt__":
                yield {"type": "pending", "payload": upd[0].value if isinstance(upd, (list, tuple)) else None}
                continue
            if node == "input_guard" and upd.get("guard_notice"):
                yield {"type": "notice", "text": upd["guard_notice"]}
            if node == "supervisor" and upd.get("next_agent"):
                yield {"type": "route", "agent": upd["next_agent"]}
            for m in upd.get("messages", []):
                if m.type == "ai" and m.content:
                    yield {"type": "reply", "node": node, "text": str(m.content)}


def _paused_thread(graph, refund: dict) -> tuple[dict, dict] | None:
    """(customer, cfg) of the conversation actually paused on this refund, or None."""
    customer = identity.get_customer(refund["customer_id"])
    if customer is None:
        return None
    for surface in SURFACES:
        cfg = session_config(customer, surface=surface)
        waiting = pending_review(graph, cfg)
        if waiting and waiting.get("refund_id") == refund["refund_id"]:
            return customer, cfg
    return None


def pending_reviews(graph) -> list[dict]:
    """Refunds awaiting a decision whose conversation is paused and can be resumed (oldest first).

    A pending row with no paused conversation (e.g. the checkpoint was lost) is skipped with a
    warning: approving it here would pay it without telling the customer anything.
    """
    found = []
    for refund in pending_refunds():
        located = _paused_thread(graph, refund)
        if located is None:
            log.warning("pending refund %s has no paused conversation; not listed", refund["refund_id"])
            continue
        found.append({**refund, "thread_id": located[1]["configurable"]["thread_id"]})
    return found


def resolve_review(graph, refund_id: int, approved: bool, note: str = "") -> dict:
    """Record the reviewer's decision and let the customer's turn finish.

    Returns {"status": "decided", "message": <what the customer is told>} or
    {"status": "already_decided"} if nobody is waiting on that refund any more (another
    reviewer got there first). The decision itself is applied by `decide_refund`, which is idempotent.
    """
    refund = next((r for r in pending_refunds() if r["refund_id"] == refund_id), None)
    located = _paused_thread(graph, refund) if refund else None
    if located is None:
        return {"status": "already_decided"}
    _, cfg = located
    message = ""
    for event in turn_events(graph, Command(resume={"approved": bool(approved), "note": note.strip()}), cfg):
        if event["type"] == "reply":
            message = event["text"]
    return {"status": "decided", "message": message}
