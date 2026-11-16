"""Human-in-the-loop approval node for refunds over the approval threshold.

The refund agent queues a `pending_approval` row; this node pauses the graph with
`interrupt()` until a human decides. LangGraph re-runs a resumed node from its first
line, so everything before `interrupt()` must be safe to repeat (it only reads), and
the decision itself is applied by idempotent plain code, not by an LLM.
"""
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from ecommerce_agents.state import AgentState
from ecommerce_agents.tools.refund_tools import decide_refund, get_refund


def approval_node(state: AgentState) -> dict:
    refund_id = state["pending_refund_id"]
    refund = get_refund(refund_id)
    # Pauses here. Resume value: {"approved": bool, "note": str}
    decision = interrupt({
        "refund_id": refund_id,
        "order_id": refund["order_id"],
        "amount": refund["amount"],
        "reason": refund["reason"],
    })
    result = decide_refund(refund_id, bool(decision.get("approved")), decision.get("note", ""))
    return {"pending_refund_id": None,
            "messages": [AIMessage(content=_customer_message(result, refund), name="approval")]}


def _customer_message(result: dict, refund: dict) -> str:
    final = result.get("refund", refund)
    amount, order_id = final["amount"], final["order_id"]
    if result.get("already_decided"):
        return f"The refund request for order {order_id} has already been reviewed ({final['status']})."
    if result["success"]:
        return f"Good news: your refund of ${amount:.2f} for order {order_id} was approved and processed."
    note = f" Reason: {final['reviewer_note']}" if final.get("reviewer_note") else ""
    return (f"Your refund request for order {order_id} was reviewed and not approved.{note} "
            "This decision is final.")
