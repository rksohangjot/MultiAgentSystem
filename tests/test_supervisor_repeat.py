"""A specialist runs at most once per customer message (code-enforced), and the loop cap escalates.

Regression for two bugs the eval judge found: after an approved over-$50 refund the supervisor sent
the request back to the refund agent, whose stale "queued for review" reply overwrote the approval
outcome; and the loop-cap handoff did not mark the turn escalated.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from ecommerce_agents import supervisor, tracing
from ecommerce_agents.supervisor import RouteDecision, specialists_run_this_turn, supervisor_node
from ecommerce_agents.graph import build_graph
from ecommerce_agents.tools import refund_tools as rt
from tests.test_refund_graph import (CFG, big_refund_model, frozen_today, seeded, stub_models,  # noqa: F401
                                     stub_supervisor)


def ai(name, text="answer"):
    return AIMessage(content=text, name=name)


def test_only_messages_after_the_latest_customer_message_count():
    msgs = [HumanMessage("refund 1011"), ai("refund"), HumanMessage("and where is 1002?"), ai("order_status")]
    assert specialists_run_this_turn(msgs) == {"order_status"}  # last turn's refund is ignored
    assert specialists_run_this_turn([HumanMessage("hi")]) == set()


def test_non_specialist_messages_are_not_counted():
    msgs = [HumanMessage("x"), ai("approval"), ai("input_guard"), AIMessage(content="no name")]
    assert specialists_run_this_turn(msgs) == set()


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(tracing, "record_event", lambda name, **detail: seen.append((name, detail)))
    return seen


def decide_as(monkeypatch, agent):
    monkeypatch.setattr(supervisor, "decide", lambda m: RouteDecision(next_agent=agent))


def test_repeat_of_a_specialist_that_already_answered_is_blocked(monkeypatch, events):
    decide_as(monkeypatch, "refund")
    out = supervisor_node({"messages": [HumanMessage("refund 1011"), ai("refund")], "iterations": 1})
    assert out["next_agent"] == "FINISH" and "messages" not in out
    assert events == [("supervisor.repeat_blocked", {"agent": "refund"})]


def test_same_specialist_is_allowed_again_on_a_new_customer_message(monkeypatch, events):
    decide_as(monkeypatch, "refund")
    msgs = [HumanMessage("refund 1011"), ai("refund"), HumanMessage("now refund 1008")]
    assert supervisor_node({"messages": msgs, "iterations": 0})["next_agent"] == "refund"
    assert events == []


def test_a_different_specialist_is_still_allowed_for_mixed_requests(monkeypatch, events):
    decide_as(monkeypatch, "order_status")
    out = supervisor_node({"messages": [HumanMessage("late and refund"), ai("refund")], "iterations": 1})
    assert out["next_agent"] == "order_status" and events == []


def test_loop_cap_marks_the_turn_escalated(monkeypatch, events):
    out = supervisor_node({"messages": [HumanMessage("hi")], "iterations": supervisor.MAX_ITERATIONS})
    assert out["next_agent"] == "FINISH" and out["escalated"] is True
    assert out["messages"][0].content == supervisor.CAP_REPLY
    assert events == [("escalation.loop_cap", {"node": "supervisor"})]


def test_approved_refund_final_reply_is_the_approval_outcome_not_a_stale_one(monkeypatch, frozen_today):
    # The model keeps asking for "refund": before the fix the refund agent ran again after approval
    # (its script has only one run, so a second run would also crash the model stub).
    stub_supervisor(monkeypatch, ["refund", "refund", "refund"])
    stub_models(monkeypatch, refund_model=big_refund_model())
    g = build_graph()
    out = g.invoke({"messages": [HumanMessage("Refund order 1011"), ], "iterations": 0}, CFG)
    rid = out["__interrupt__"][0].value["refund_id"]
    out = g.invoke(Command(resume={"approved": True, "note": "ok"}), CFG)
    last = out["messages"][-1]
    assert rt.get_refund(rid)["status"] == "completed"
    assert last.name == "approval" and "approved and processed" in last.content
    assert "queued" not in last.content
    refund_replies = [m for m in out["messages"] if m.type == "ai" and m.name == "refund" and m.content]
    assert len(refund_replies) == 1  # one run (tool-call messages have no text); no stale second reply
