from langgraph.prebuilt import create_react_agent

from ecommerce_agents.llm import get_llm
from ecommerce_agents.tools.product_tools import PRODUCT_TOOLS

PROMPT = """You are the Product-Info agent for an online store.
Use your tools to answer questions about products: specs, price, stock and
compatibility. Only state facts returned by tools; if something isn't in the
data, say you don't know. Be concise."""


def build_product_agent():
    return create_react_agent(get_llm(), PRODUCT_TOOLS, prompt=PROMPT, name="product_info")
