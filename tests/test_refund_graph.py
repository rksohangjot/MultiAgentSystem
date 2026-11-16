"""End-to-end graph tests: stubbed supervisor + scripted models (no API calls)."""
import pytest
from langchain_core.messages import HumanMessage

from ecommerce_agents import db, supervisor
from ecommerce_agents.agents import order_status, product_info, refund
from ecommerce_agents.graph import build_graph
from ecommerce_agents.supervisor import RouteDecision
from tests.test_refund_agent import ScriptedModel, call
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def seeded(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.seed(path)


def stub_supervisor(monkeypatch, decisions):
    it = iter(decisions)
    monkeypatch.setattr(supervisor, "decide", lambda m: RouteDecision(next_agent=next(it)))


def stub_models(monkeypatch, *, order=None, refund_model=None):
    idle = ScriptedModel(script=[AIMessage(content="unused")])
    monkeypatch.setattr(order_status, "get_llm", lambda *a, **k: order or idle)
    monkeypatch.setattr(product_info, "get_llm", lambda *a, **k: idle)
    monkeypatch.setattr(refund, "get_llm", lambda *a, **k: refund_model or idle)


def run(text):
    cfg = {"configurable": {"thread_id": "t"}, "recursion_limit": 25}
    return build_graph().invoke({"messages": [HumanMessage(text)], "iterations": 0}, cfg)


def refund_ids():
    conn = db.get_conn()
    try:
        return [r["order_id"] for r in conn.execute("SELECT order_id FROM refunds ORDER BY id")]
    finally:
        conn.close()


def test_route_maps_refund():
    assert supervisor.route({"next_agent": "refund"}) == "refund"


def test_supervisor_routes_to_refund_and_refund_is_issued(monkeypatch):
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    stub_models(monkeypatch, refund_model=ScriptedModel(script=[
        call("check_return_eligibility", {"order_id": 1008}, "a"),
        call("initiate_refund", {"order_id": 1008, "reason": "broken"}, "b"),
        AIMessage(content="Refund issued for order 1008."),
    ]))
    out = run("Refund order 1008, it is broken")
    assert out["messages"][-1].content == "Refund issued for order 1008."
    assert refund_ids() == [1010, 1008]


def test_mixed_request_goes_order_status_then_refund(monkeypatch):
    stub_supervisor(monkeypatch, ["order_status", "refund", "FINISH"])
    stub_models(
        monkeypatch,
        order=ScriptedModel(script=[call("get_order_status", {"order_id": 1008}, "o"),
                                    AIMessage(content="Order 1008 was delivered.")]),
        refund_model=ScriptedModel(script=[call("check_return_eligibility", {"order_id": 1008}, "c"),
                                           AIMessage(content="Order 1008 is eligible for a refund.")]),
    )
    out = run("Where is order 1008 and can I get a refund?")
    names = [m.name for m in out["messages"] if m.type == "ai" and m.content]
    assert names == ["order_status", "refund"]
    assert out["iterations"] == 3  # supervisor ran 3 times, under the cap of 4


def test_refund_handoff_message_reaches_user_and_graph_finishes(monkeypatch):
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    stub_models(monkeypatch, refund_model=ScriptedModel(
        script=[call("check_return_eligibility", {"order_id": 1008}, "x")], loop=True))
    out = run("refund 1008")
    assert out["messages"][-1].content == refund.HANDOFF
    assert refund_ids() == [1010]


# --- human-in-the-loop approval (refunds over $50) ---
import datetime as _dt

from langgraph.types import Command

from ecommerce_agents.tools import refund_tools as rt


@pytest.fixture
def frozen_today(monkeypatch):
    """Pin 'today' so order 1011 (delivered 2026-11-08) stays inside the 30-day window."""
    class Frozen(_dt.date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 13)
    monkeypatch.setattr(rt, "date", Frozen)


def big_refund_model():
    return ScriptedModel(script=[
        call("check_return_eligibility", {"order_id": 1011}, "a"),
        call("initiate_refund", {"order_id": 1011, "reason": "changed mind"}, "b"),
        AIMessage(content="A human agent will review your refund."),
    ])


CFG = {"configurable": {"thread_id": "hitl"}, "recursion_limit": 25}


def start_big_refund(monkeypatch, graph_decisions):
    stub_supervisor(monkeypatch, graph_decisions)
    stub_models(monkeypatch, refund_model=big_refund_model())
    g = build_graph()
    out = g.invoke({"messages": [HumanMessage("Refund order 1011")], "iterations": 0}, CFG)
    return g, out


def test_over_50_pauses_for_human_and_pays_nothing(monkeypatch, frozen_today):
    g, out = start_big_refund(monkeypatch, ["refund", "FINISH"])
    payload = out["__interrupt__"][0].value
    assert payload["order_id"] == 1011 and payload["amount"] == 89.0
    assert rt.get_refund(payload["refund_id"])["status"] == "pending_approval"
    assert g.get_state(CFG).next == ("approval",)  # paused at the approval node


def test_human_approves_then_customer_is_told(monkeypatch, frozen_today):
    g, out = start_big_refund(monkeypatch, ["refund", "FINISH"])
    rid = out["__interrupt__"][0].value["refund_id"]
    out = g.invoke(Command(resume={"approved": True, "note": "ok"}), CFG)
    assert rt.get_refund(rid)["status"] == "completed"
    assert "approved" in out["messages"][-1].content
    assert g.get_state(CFG).next == ()  # graph finished


def test_human_rejects_then_customer_is_told(monkeypatch, frozen_today):
    g, out = start_big_refund(monkeypatch, ["refund", "FINISH"])
    rid = out["__interrupt__"][0].value["refund_id"]
    out = g.invoke(Command(resume={"approved": False, "note": "item used"}), CFG)
    assert rt.get_refund(rid)["status"] == "rejected"
    assert "not approved" in out["messages"][-1].content and "item used" in out["messages"][-1].content


def test_under_50_never_reaches_approval(monkeypatch):
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    stub_models(monkeypatch, refund_model=ScriptedModel(script=[
        call("initiate_refund", {"order_id": 1008, "reason": "broken"}, "b"),
        AIMessage(content="Refund issued."),
    ]))
    out = run("Refund order 1008")
    assert "__interrupt__" not in out and refund_ids() == [1010, 1008]


def test_pause_survives_restart_with_sqlite_checkpointer(monkeypatch, frozen_today, tmp_path):
    from ecommerce_agents.graph import sqlite_checkpointer
    db_file = tmp_path / "cp.db"
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    stub_models(monkeypatch, refund_model=big_refund_model())
    g1 = build_graph(sqlite_checkpointer(db_file))
    out = g1.invoke({"messages": [HumanMessage("Refund order 1011")], "iterations": 0}, CFG)
    rid = out["__interrupt__"][0].value["refund_id"]
    # "Restart": brand-new graph and checkpointer objects over the same file.
    stub_supervisor(monkeypatch, ["FINISH"])
    g2 = build_graph(sqlite_checkpointer(db_file))
    out = g2.invoke(Command(resume={"approved": True, "note": ""}), CFG)
    assert rt.get_refund(rid)["status"] == "completed" and "approved" in out["messages"][-1].content
