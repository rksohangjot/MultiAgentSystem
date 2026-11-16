"""Approval node tests: pause with interrupt(), resume with a human decision."""
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ecommerce_agents import db
from ecommerce_agents.approval import approval_node
from ecommerce_agents.state import AgentState
from ecommerce_agents.tools import refund_tools as rt

CFG = {"configurable": {"thread_id": "a"}}


@pytest.fixture(autouse=True)
def seeded(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.seed(path)


def mini_graph():
    g = StateGraph(AgentState)
    g.add_node("approval", approval_node)
    g.add_edge(START, "approval")
    g.add_edge("approval", END)
    return g.compile(checkpointer=MemorySaver())


def start():
    # Queue a >$50 refund on a fixed day (order 1011 would be out of window on the real date)
    rid = rt.create_refund(1011, "changed mind", rt.date(2026, 11, 13))["refund"]["id"]
    return rid, mini_graph()


def test_pauses_with_review_details_and_does_not_pay():
    rid, g = start()
    out = g.invoke({"messages": [], "pending_refund_id": rid}, CFG)
    payload = out["__interrupt__"][0].value
    assert payload == {"refund_id": rid, "order_id": 1011, "amount": 89.0, "reason": "changed mind"}
    assert rt.get_refund(rid)["status"] == "pending_approval"


def test_resume_approve(monkeypatch):
    rid, g = start()
    g.invoke({"messages": [], "pending_refund_id": rid}, CFG)
    # approval re-checks the 30-day window against today, so pin "today" for determinism
    monkeypatch.setattr("ecommerce_agents.approval.decide_refund",
                        lambda i, a, n="": rt.decide_refund(i, a, n, rt.date(2026, 11, 13)))
    out = g.invoke(Command(resume={"approved": True, "note": "ok"}), CFG)
    assert rt.get_refund(rid)["status"] == "completed"
    assert out["pending_refund_id"] is None
    assert "approved" in out["messages"][-1].content and "$89.00" in out["messages"][-1].content


def test_resume_reject(monkeypatch):
    rid, g = start()
    g.invoke({"messages": [], "pending_refund_id": rid}, CFG)
    out = g.invoke(Command(resume={"approved": False, "note": "item used"}), CFG)
    assert rt.get_refund(rid)["status"] == "rejected"
    msg = out["messages"][-1].content
    assert "not approved" in msg and "item used" in msg and "final" in msg
    assert out["pending_refund_id"] is None
