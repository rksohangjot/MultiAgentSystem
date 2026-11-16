"""Chat-loop logic shared by the CLI: stream a turn, and handle human-approval pauses."""
from langchain_core.messages import HumanMessage
from langgraph.types import Command


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


def new_message(text: str) -> dict:
    # Reset the per-turn loop counter; messages persist via the checkpointer.
    return {"messages": [HumanMessage(text)], "iterations": 0}


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
