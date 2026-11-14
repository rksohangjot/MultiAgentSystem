from langgraph.prebuilt import create_react_agent

from ecommerce_agents.llm import get_llm
from ecommerce_agents.tools import AGENT_TOOLS

PROMPT = """You are the Order-Status agent for an online store.
Use your tools to look up orders. Only answer questions about order status,
tracking and delivery. If no order ID is given, ask for it. Never guess order
details; if the tool returns an error, say so. Be concise."""


def build_order_agent():
    return create_react_agent(get_llm(), AGENT_TOOLS["order_status"], prompt=PROMPT, name="order_status")
