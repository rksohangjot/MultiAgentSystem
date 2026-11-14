"""Refund agent: the only agent with a write tool (initiate_refund)."""
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from ecommerce_agents.llm import get_llm
from ecommerce_agents.tools import AGENT_TOOLS

# Hard cap on the ReAct loop (loop prevention). ~3 tool round-trips + final answer.
RECURSION_LIMIT = 8

# Least privilege: order lookup (read-only) + refund tools. No product search.
REFUND_AGENT_TOOLS = AGENT_TOOLS["refund"]

PROMPT = """You are the Returns/Refund agent for an online store.

Process, in this order:
1. If no order ID is given, ask for it. Never guess an order ID.
2. Call check_return_eligibility for the order before anything else.
3. If the customer only asks whether they CAN return/refund, explain the result and stop.
   Do not issue a refund.
4. If the customer explicitly asks for a refund and the order is eligible, call
   initiate_refund. Use the customer's stated reason, or "customer request" if none.

Rules:
- Never say a refund was issued unless initiate_refund returned success: true.
- If initiate_refund returns needs_human_approval, tell the customer a human agent will
  review it. Do not promise the refund.
- If the order is not eligible, state the policy reason plainly. Do not offer workarounds.
- Use get_return_policy if the customer asks about the rules.
- Be concise and only state facts returned by tools."""

HANDOFF = "Sorry, I couldn't complete that refund request. Let me get a human to help."
# create_react_agent returns this text (instead of raising) when it runs out of steps.
OUT_OF_STEPS_PREFIX = "Sorry, need more steps"


def build_refund_agent():
    """Return a graph node that runs the ReAct agent with a loop cap and human handoff.

    The cap can surface two ways: the agent's own out-of-steps message, or a
    GraphRecursionError. Both are turned into the same human-handoff reply.
    """
    agent = create_react_agent(get_llm(), REFUND_AGENT_TOOLS, prompt=PROMPT, name="refund")

    def refund_node(state, config):
        try:
            result = agent.invoke(state, {**config, "recursion_limit": RECURSION_LIMIT})
        except GraphRecursionError:
            return {"messages": [AIMessage(content=HANDOFF, name="refund")]}
        if str(result["messages"][-1].content).startswith(OUT_OF_STEPS_PREFIX):
            return {"messages": [AIMessage(content=HANDOFF, name="refund")]}
        return {"messages": result["messages"]}

    return refund_node
