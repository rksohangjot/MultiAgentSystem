"""Supervisor tests with a stubbed router (no API calls) + optional live routing tests."""
import os

import pytest
from langchain_core.messages import HumanMessage

from ecommerce_agents import supervisor
from ecommerce_agents.supervisor import RouteDecision, supervisor_node


def test_iteration_cap_forces_finish():
    out = supervisor_node({"messages": [HumanMessage("hi")], "iterations": supervisor.MAX_ITERATIONS})
    assert out["next_agent"] == "FINISH"


def test_supervisor_replies_directly_on_greeting(monkeypatch):
    monkeypatch.setattr(supervisor, "decide", lambda m: RouteDecision(next_agent="FINISH", reply="Hello!"))
    out = supervisor_node({"messages": [HumanMessage("hi")], "iterations": 0})
    assert out["next_agent"] == "FINISH" and out["messages"][0].content == "Hello!"


def test_route_maps_finish_to_end():
    assert supervisor.route({"next_agent": "FINISH"}) == "__end__"
    assert supervisor.route({"next_agent": "order_status"}) == "order_status"


needs_key = pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")


@pytest.mark.llm
@needs_key
@pytest.mark.parametrize("text,expected", [
    ("Where is my order 1002?", "order_status"),
    ("Does the rain jacket come in blue?", "product_info"),
    ("I want to return order 1003", "refund"),
    ("Can I get a refund for order 1004?", "refund"),
])
def test_live_routing(text, expected):
    assert supervisor.decide([HumanMessage(text)]).next_agent == expected


@pytest.mark.llm
@needs_key
def test_live_mixed_message_visits_both_agents():
    from ecommerce_agents.graph import build_graph

    graph = build_graph()
    cfg = {"configurable": {"thread_id": "mixed-test"}, "recursion_limit": 40}
    visited = set()
    for update in graph.stream(
        {"messages": [HumanMessage("My order 1002 is late and I want my money back")]},
        cfg, stream_mode="updates",
    ):
        visited.update(update.keys())
    assert {"order_status", "refund"} <= visited
