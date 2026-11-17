"""Wire supervisor + agents into one LangGraph."""
import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from ecommerce_agents.approval import approval_node
from ecommerce_agents.agents.order_status import build_order_agent
from ecommerce_agents.agents.product_info import build_product_agent
from ecommerce_agents.agents.refund import build_refund_agent
from ecommerce_agents.input_guard import input_guard_node, route_after_guard
from ecommerce_agents.output_guard import guarded
from ecommerce_agents.state import AgentState
from ecommerce_agents.supervisor import route, supervisor_node


def sqlite_checkpointer(path: Path | str) -> SqliteSaver:
    """Durable checkpointer: a refund waiting for approval survives a process restart."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def route_after_refund(state: AgentState) -> str:
    """A refund queued for human approval goes to the approval node; otherwise back to the supervisor."""
    return "approval" if state.get("pending_refund_id") else "supervisor"


def _agent_node(agent):
    """Run a compiled ReAct agent as a plain node (only `messages` flows back to the parent)."""
    return lambda state, config: {"messages": agent.invoke(state, config)["messages"]}


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("input_guard", input_guard_node)
    # Every node that writes customer-visible text is wrapped by the output guard.
    # (The approval node is not: its text is built by code from the database.)
    g.add_node("supervisor", guarded(lambda state, config: supervisor_node(state)))
    g.add_node("order_status", guarded(_agent_node(build_order_agent())))
    g.add_node("product_info", guarded(_agent_node(build_product_agent())))
    g.add_node("refund", guarded(build_refund_agent()))
    g.add_node("approval", approval_node)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_guard, {"supervisor": "supervisor", "__end__": END})
    g.add_conditional_edges("supervisor", route,
                            {"order_status": "order_status", "product_info": "product_info",
                             "refund": "refund", "__end__": END})
    # Specialists always hand control back to the supervisor.
    g.add_edge("order_status", "supervisor")
    g.add_edge("product_info", "supervisor")
    g.add_conditional_edges("refund", route_after_refund,
                            {"approval": "approval", "supervisor": "supervisor"})
    g.add_edge("approval", "supervisor")
    return g.compile(checkpointer=checkpointer or MemorySaver())
