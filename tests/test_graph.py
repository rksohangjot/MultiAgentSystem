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
])
def test_live_routing(text, expected):
    assert supervisor.decide([HumanMessage(text)]).next_agent == expected
