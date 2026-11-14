"""Wire supervisor + agents into one LangGraph."""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ecommerce_agents.agents.order_status import build_order_agent
from ecommerce_agents.agents.product_info import build_product_agent
from ecommerce_agents.state import AgentState
from ecommerce_agents.supervisor import route, supervisor_node


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("order_status", build_order_agent())
    g.add_node("product_info", build_product_agent())
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route,
                            {"order_status": "order_status", "product_info": "product_info", "__end__": END})
    # Specialists always hand control back to the supervisor.
    g.add_edge("order_status", "supervisor")
    g.add_edge("product_info", "supervisor")
    return g.compile(checkpointer=checkpointer or MemorySaver())
