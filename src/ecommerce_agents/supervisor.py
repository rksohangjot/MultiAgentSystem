"""Supervisor: decides which specialist handles the conversation next."""
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ecommerce_agents import tracing
from ecommerce_agents.llm import get_llm
from ecommerce_agents.state import AgentState

MAX_ITERATIONS = 4  # hard cap on supervisor turns: loop prevention
SPECIALISTS = ("order_status", "product_info", "refund")
CAP_REPLY = "Sorry, I couldn't complete that. Let me get a human to help."

SYSTEM = """You are the supervisor of an e-commerce support team. Pick who acts next:
- order_status: questions about an order's status, tracking, delivery.
- product_info: questions about products (specs, price, stock, compatibility).
- refund: whether an order can be returned/refunded, and requests to refund an order.
- FINISH: the customer's request has been fully answered by a specialist, or the
  message is a greeting/off-topic (then reply directly in `reply`).
If a specialist has already answered every part of the user's request, choose FINISH.
Never choose a specialist whose answer is already in the conversation."""


class RouteDecision(BaseModel):
    next_agent: Literal["order_status", "product_info", "refund", "FINISH"] = Field(
        description="Who should act next.")
    reply: str = Field(default="", description="Direct reply, only when FINISH with no specialist answer.")


def decide(messages) -> RouteDecision:
    llm = get_llm().with_structured_output(RouteDecision)
    return llm.invoke([SystemMessage(content=SYSTEM), *messages])


def specialists_run_this_turn(messages) -> set[str]:
    """Specialists that already answered since the customer's latest message (read from message names)."""
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return {m.name for m in messages[start + 1:] if isinstance(m, AIMessage) and m.name in SPECIALISTS}


def supervisor_node(state: AgentState) -> dict:
    iterations = state.get("iterations", 0) + 1
    if iterations > MAX_ITERATIONS:
        tracing.record_event("escalation.loop_cap", node="supervisor")
        return {"next_agent": "FINISH", "iterations": iterations, "escalated": True,
                "messages": [AIMessage(content=CAP_REPLY)]}
    decision = decide(state["messages"])
    if decision.next_agent in specialists_run_this_turn(state["messages"]):
        # Code-enforced: the prompt asks the model not to repeat a specialist, but it still does
        # (e.g. re-sending a finished refund to the refund agent, whose stale reply then
        # overwrites the approval outcome). A specialist runs at most once per customer message.
        tracing.record_event("supervisor.repeat_blocked", agent=decision.next_agent)
        decision = RouteDecision(next_agent="FINISH")
    update: dict = {"next_agent": decision.next_agent, "iterations": iterations}
    last = state["messages"][-1]
    # Only speak directly if no specialist has just answered.
    if decision.next_agent == "FINISH" and decision.reply and not isinstance(last, AIMessage):
        update["messages"] = [AIMessage(content=decision.reply, name="supervisor")]
    return update


def route(state: AgentState) -> Literal["order_status", "product_info", "refund", "__end__"]:
    return "__end__" if state["next_agent"] == "FINISH" else state["next_agent"]
