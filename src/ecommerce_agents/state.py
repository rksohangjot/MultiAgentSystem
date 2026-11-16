"""Shared graph state. Agents communicate only through this."""
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    next_agent: str   # supervisor's routing decision
    iterations: int   # supervisor turns this request (loop prevention)
    pending_refund_id: int | None  # refund awaiting human approval (set by refund agent, cleared by approval node)
