"""Chat-loop logic shared by the CLI: stream a turn, and handle human-approval pauses."""
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ecommerce_agents import guards, identity, tracing

MAX_SIGN_IN_ATTEMPTS = 3


def stream_turn(graph, graph_input, cfg, review, emit=print) -> None:
    """Run one turn. Each time the graph pauses for human approval, ask `review` and resume.

    `review(payload) -> {"approved": bool, "note": str}` plays the human reviewer.
    """
    while True:
        paused = None
        for step in graph.stream(graph_input, cfg, stream_mode="updates"):
            for node, upd in step.items():
                if node == "__interrupt__":
                    paused = upd[0].value
                    continue
                if node == "input_guard" and (upd or {}).get("guard_notice"):
                    emit(f"  [notice] {upd['guard_notice']}")
                if node == "supervisor":
                    emit(f"  [supervisor -> {(upd or {}).get('next_agent')}]")
                for m in (upd or {}).get("messages", []):
                    if m.type == "ai" and m.content:
                        emit(f"{node}> {m.content}")
        if paused is None:
            return
        graph_input = Command(resume=review(paused))


def pending_review(graph, cfg) -> dict | None:
    """Payload of an approval left waiting (e.g. before a restart), or None."""
    for task in graph.get_state(cfg).tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def resume_pending(graph, cfg, review, emit=print) -> bool:
    """If a refund is waiting for a decision, collect it and finish that turn."""
    if pending_review(graph, cfg) is None:
        return False
    emit("A refund from an earlier session is waiting for human review.")
    stream_turn(graph, None, cfg, review, emit)
    return True


def new_message(text: str, customer: dict | None = None) -> dict:
    """Input for one turn. Resets the per-turn loop counter.

    PII is redacted here too, before the text enters the graph: LangGraph writes the raw
    input to the checkpoint before any node runs, so the node alone cannot keep it out.
    (The graph's input_guard node still redacts, as the check for every other entry point.)
    """
    own = customer["email"] if customer else None
    return {"messages": [HumanMessage(guards.redact_pii(text, own_email=own).text)], "iterations": 0}


def cli_review(payload: dict, ask=input, emit=print) -> dict:
    """Terminal reviewer: show the request, ask approve/reject."""
    emit(f"\n  [HUMAN REVIEW] Refund #{payload['refund_id']}: order {payload['order_id']}, "
         f"${payload['amount']:.2f}, reason: {payload['reason']!r}")
    while True:
        answer = ask("  approve? (y/n): ").strip().lower()
        if answer in {"y", "yes"}:
            return {"approved": True, "note": ask("  note (optional): ").strip()}
        if answer in {"n", "no"}:
            return {"approved": False, "note": ask("  reason for rejecting: ").strip()}


def sign_in(ask=input, emit=print) -> dict | None:
    """Stand-in login: ask for an email, return the customer row or None after 3 misses.

    Real authentication arrives with the Streamlit UI; the rest of the system only
    needs the resulting customer id.
    """
    for _ in range(MAX_SIGN_IN_ATTEMPTS):
        customer = identity.find_customer_by_email(ask("Email: "))
        if customer:
            emit(f"Signed in as {customer['name']}.")
            return customer
        emit("No account with that email.")
    return None


def session_config(customer: dict) -> dict:
    """Run config for one customer. The thread is per customer so chats (and pending
    approvals) are never shared; customer_id is read by the tools, never by the LLM.

    When Langfuse is configured, the handler goes in `callbacks` and the session/user ids in
    `metadata`, so every turn of a customer's thread groups under one Langfuse session
    (a resume after an approval pause is a separate trace in the same session)."""
    thread_id = f"cli-{customer['id']}"
    cfg = {"configurable": {"thread_id": thread_id, "customer_id": customer["id"]},
           "recursion_limit": 25}
    handlers = tracing.callbacks()
    if handlers:
        cfg["callbacks"] = handlers
        cfg["metadata"] = {"langfuse_session_id": thread_id,
                           "langfuse_user_id": f"customer-{customer['id']}",
                           "langfuse_tags": ["cli", "ecommerce-agents"]}
    return cfg
