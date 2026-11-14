"""Supervisor: decides which specialist handles the conversation next."""
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from pydantic import BaseModel, Field

from ecommerce_agents.llm import get_llm
from ecommerce_agents.state import AgentState

MAX_ITERATIONS = 4  # hard cap on supervisor turns: loop prevention

SYSTEM = """You are the supervisor of an e-commerce support team. Pick who acts next:
- order_status: questions about an order's status, tracking, delivery.
- product_info: questions about products (specs, price, stock, compatibility).
- FINISH: the customer's request has been fully answered by a specialist, or the
  message is a greeting/off-topic (then reply directly in `reply`).
If a specialist has already answered every part of the user's request, choose FINISH.
Never choose a specialist whose answer is already in the conversation."""


class RouteDecision(BaseModel):
    next_agent: Literal["order_status", "product_info", "FINISH"] = Field(
        description="Who should act next.")
    reply: str = Field(default="", description="Direct reply, only when FINISH with no specialist answer.")


def decide(messages) -> RouteDecision:
    llm = get_llm().with_structured_output(RouteDecision)
    return llm.invoke([SystemMessage(content=SYSTEM), *messages])


def supervisor_node(state: AgentState) -> dict:
    iterations = state.get("iterations", 0) + 1
    if iterations > MAX_ITERATIONS:
        return {"next_agent": "FINISH", "iterations": iterations,
                "messages": [AIMessage(content="Sorry, I couldn't complete that. Let me get a human to help.")]}
    decision = decide(state["messages"])
    update: dict = {"next_agent": decision.next_agent, "iterations": iterations}
    last = state["messages"][-1]
    # Only speak directly if no specialist has just answered.
    if decision.next_agent == "FINISH" and decision.reply and not isinstance(last, AIMessage):
        update["messages"] = [AIMessage(content=decision.reply, name="supervisor")]
    return update


def route(state: AgentState) -> Literal["order_status", "product_info", "__end__"]:
    return "__end__" if state["next_agent"] == "FINISH" else state["next_agent"]
