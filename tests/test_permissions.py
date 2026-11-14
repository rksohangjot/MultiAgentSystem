"""Least-privilege checks: only the refund agent may hold a write tool."""
from ecommerce_agents.tools import AGENT_TOOLS, WRITE_TOOLS


def names(agent):
    return [t.name for t in AGENT_TOOLS[agent]]


def test_only_refund_agent_has_write_tools():
    for agent in AGENT_TOOLS:
        holds_write = bool(WRITE_TOOLS & set(names(agent)))
        assert holds_write == (agent == "refund"), agent


def test_every_write_tool_has_exactly_one_owner():
    for tool in WRITE_TOOLS:
        owners = [a for a in AGENT_TOOLS if tool in names(a)]
        assert owners == ["refund"]


def test_read_agents_tool_lists():
    assert names("order_status") == ["get_order_status"]
    assert names("product_info") == ["search_products"]
    assert "search_products" not in names("refund")
