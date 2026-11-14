"""Per-agent tool allowlists (least privilege). Only the refund agent may write."""
from ecommerce_agents.tools.order_tools import ORDER_TOOLS, get_order_status
from ecommerce_agents.tools.product_tools import PRODUCT_TOOLS
from ecommerce_agents.tools.refund_tools import REFUND_TOOLS

# Tools that change data. Anything listed here must belong to exactly one agent.
WRITE_TOOLS = {"initiate_refund"}

AGENT_TOOLS = {
    "order_status": ORDER_TOOLS,
    "product_info": PRODUCT_TOOLS,
    "refund": [get_order_status, *REFUND_TOOLS],
}
