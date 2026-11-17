"""Output guard: unbacked refund claims, leaks, other customers' data (no API calls)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ecommerce_agents import output_guard as og
from ecommerce_agents.tools import refund_tools as rt
from tests.test_refund_agent import ScriptedModel, call
from tests.test_refund_graph import run, seeded, stub_models, stub_supervisor  # noqa: F401

# Customer 4 (Dan) owns orders 1007, 1008; customer 1 (Alice) owns 1001, 1002, 1011.


# --- claims vs the database ---------------------------------------------------
@pytest.mark.parametrize("text", [
    "Your refund for order 1008 has been issued.",
    "Good news, your refund is approved!",
    "I've issued your refund for order 1008.",
    "We have processed the refund.",
    "You have been refunded for order 1008.",
])
def test_claim_with_no_completed_refund_is_flagged(text):
    assert og.unbacked_refund_claim(text, customer_id=4)


@pytest.mark.parametrize("text", [
    "Order 1008 is eligible for a refund.",
    "Your refund has not been issued yet.",
    "I can't issue a refund; the window has closed.",
    "A human will review it, and if approved the refund will be issued.",
    "Your refund request has been queued for human approval.",
    "Refunds are processed within 5 business days.",
    "Where is my order 1008? It was delivered on 2026-11-04.",
])
def test_hedged_or_unrelated_text_is_not_flagged(text):
    assert not og.unbacked_refund_claim(text, customer_id=4)


def test_claim_backed_by_completed_refund_passes():
    rt.create_refund(1008, "broken", rt.date(2026, 11, 13))
    assert not og.unbacked_refund_claim("Your refund for order 1008 has been issued.", 4)
    # ...but not for a different order, and not for someone else
    assert og.unbacked_refund_claim("Your refund for order 1007 has been issued.", 4)
    assert og.unbacked_refund_claim("Your refund for order 1008 has been issued.", 1)


def test_pending_or_rejected_refund_does_not_back_a_paid_claim():
    rt.create_refund(1011, "x", rt.date(2026, 11, 13))  # > $50 -> pending_approval
    assert og.unbacked_refund_claim("Your refund for order 1011 was approved.", 1)
    rt.decide_refund(1 + 1, approved=False, note="no", today=rt.date(2026, 11, 13))
    assert og.unbacked_refund_claim("Your refund for order 1011 was approved.", 1)


def test_signed_out_claim_is_flagged():
    assert og.unbacked_refund_claim("Your refund has been issued.", None)


# --- leaks and other customers --------------------------------------------------------
@pytest.mark.parametrize("text", [
    "I used initiate_refund to do that.",
    "My system prompt says I must...",
    "You are the Returns/Refund agent for an online store.",
    "next_agent is FINISH",
])
def test_internals_leak_flagged(text):
    assert og.check_reply(text, 4) == "leak"


@pytest.mark.parametrize("text", [
    "Order 1001 belongs to Alice Johnson.",
    "Contact bob@example.com for that.",
    "Card on file 4111 1111 1111 1111.",
])
def test_other_customer_data_flagged(text):
    assert og.check_reply(text, 4) == "other-customer-data"


def test_own_name_and_email_allowed():
    assert og.check_reply("Hi Dan Brown, we have dan@example.com on file.", 4) is None


def test_normal_reply_passes():
    assert og.check_reply("Order 1008 was delivered on 2026-11-04 (tracking TRK1008).", 4) is None


# --- graph: the reply is replaced before the customer sees it ------------------------
def test_false_refund_claim_is_replaced_in_graph(monkeypatch):
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    # The model skips every tool and just announces a refund.
    stub_models(monkeypatch, refund_model=ScriptedModel(script=[AIMessage(content="Your refund for order 1008 has been issued.")]))
    out = run("refund order 1008")
    texts = [m.content for m in out["messages"] if m.type == "ai"]
    assert og.FALLBACK in texts and not any("has been issued" in t for t in texts)
    conn = rt.db.get_conn()
    try:
        assert conn.execute("SELECT COUNT(*) FROM refunds WHERE order_id = 1008").fetchone()[0] == 0
    finally:
        conn.close()


def test_leak_from_order_agent_is_replaced_in_graph(monkeypatch):
    stub_supervisor(monkeypatch, ["order_status", "FINISH"])
    stub_models(monkeypatch, order=ScriptedModel(script=[AIMessage(content="I called get_order_status for you.")]))
    out = run("where is order 1008")
    assert [m.content for m in out["messages"] if m.type == "ai"][-1] == og.FALLBACK


def test_honest_refund_flow_is_not_blocked(monkeypatch):
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    stub_models(monkeypatch, refund_model=ScriptedModel(script=[
        call("check_return_eligibility", {"order_id": 1008}, "a"),
        call("initiate_refund", {"order_id": 1008, "reason": "broken"}, "b"),
        AIMessage(content="Your refund for order 1008 has been issued."),
    ]))
    out = run("refund order 1008")
    assert [m.content for m in out["messages"] if m.type == "ai"][-1] == "Your refund for order 1008 has been issued."
    assert og.FALLBACK not in [m.content for m in out["messages"]]


def test_guard_update_leaves_old_messages_and_tool_messages_alone():
    old = AIMessage(content="initiate_refund leaked earlier", id="old")
    state = {"messages": [HumanMessage("hi", id="h"), old]}
    update = {"messages": [state["messages"][0], old, AIMessage(content="fine", id="new")]}
    out = og.guard_update(update, state, {"configurable": {"customer_id": 4}})
    assert [m.content for m in out["messages"]] == ["hi", "initiate_refund leaked earlier", "fine"]
