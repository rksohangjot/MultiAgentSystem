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
