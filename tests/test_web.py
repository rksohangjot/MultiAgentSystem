"""Web-UI logic: customer turns stop at the approval pause, a reviewer resumes the right thread."""
import pytest
from langchain_core.messages import HumanMessage

from ecommerce_agents import supervisor
from ecommerce_agents.chat_loop import new_message, session_config
from ecommerce_agents.graph import build_graph
from ecommerce_agents.supervisor import RouteDecision
from ecommerce_agents.tools import refund_tools as rt
from ecommerce_agents.web import (has_pending, history, pending_reviews, resolve_review, turn_events,
                                  web_config)
from tests.test_refund_graph import big_refund_model, frozen_today, seeded, stub_models, stub_supervisor  # noqa: F401

ALICE = {"id": 1, "name": "Alice Johnson", "email": "alice@example.com"}


def ask_for_big_refund(monkeypatch, surface="web", decisions=("refund", "FINISH")):
    """Alice asks for the $89 refund; returns (graph, cfg, events). The turn ends at the approval pause."""
    stub_supervisor(monkeypatch, list(decisions))
    stub_models(monkeypatch, refund_model=big_refund_model())
    g = build_graph()
    cfg = session_config(ALICE, surface=surface)
    events = list(turn_events(g, new_message("Refund order 1011, it is damaged", ALICE), cfg))
    return g, cfg, events


def test_web_and_cli_threads_never_mix():
    assert web_config(ALICE)["configurable"] == {"thread_id": "web-1", "customer_id": 1}
    assert session_config(ALICE)["configurable"]["thread_id"] == "cli-1"  # default unchanged


def test_turn_stops_at_the_pause_and_reports_routing_reply_and_pending(monkeypatch, frozen_today):
    g, cfg, events = ask_for_big_refund(monkeypatch)
    kinds = [e["type"] for e in events]
    assert {"type": "route", "agent": "refund"} in events
    assert "reply" in kinds and kinds[-1] == "pending"  # the turn ends waiting on a human
    assert events[-1]["payload"]["order_id"] == 1011 and events[-1]["payload"]["amount"] == 89.0
    assert has_pending(g, cfg)
    assert rt.get_refund(events[-1]["payload"]["refund_id"])["status"] == "pending_approval"  # nothing paid yet


def test_pending_reviews_lists_the_paused_refund_with_its_customer_and_thread(monkeypatch, frozen_today):
    g, cfg, _ = ask_for_big_refund(monkeypatch)
    (item,) = pending_reviews(g)
    assert item["order_id"] == 1011 and item["amount"] == 89.0 and item["customer_name"] == "Alice Johnson"
    assert item["customer_id"] == 1 and item["thread_id"] == "web-1"


def test_a_conversation_paused_in_the_cli_thread_is_found_too(monkeypatch, frozen_today):
    g, _, _ = ask_for_big_refund(monkeypatch, surface="cli")
    (item,) = pending_reviews(g)
    assert item["thread_id"] == "cli-1"


def test_nothing_pending_means_an_empty_queue(monkeypatch, frozen_today):
    assert pending_reviews(build_graph()) == []


def test_pending_row_without_a_paused_conversation_is_not_listed(frozen_today):
    rt.create_refund(1011, "x", rt.date(2026, 11, 13))  # row exists, but no graph ever paused on it
    assert rt.pending_refunds() and pending_reviews(build_graph()) == []


def test_approving_pays_the_refund_and_the_customer_sees_the_outcome(monkeypatch, frozen_today):
    g, cfg, events = ask_for_big_refund(monkeypatch)
    rid = events[-1]["payload"]["refund_id"]
    out = resolve_review(g, rid, approved=True, note="looks fine")
    assert out["status"] == "decided" and "approved and processed" in out["message"]
    assert rt.get_refund(rid)["status"] == "completed"
    assert not has_pending(g, cfg) and pending_reviews(g) == []
    assert history(g, cfg)[-1] == {"role": "assistant", "content": out["message"]}  # shown on their next refresh


def test_rejecting_records_the_note_and_tells_the_customer(monkeypatch, frozen_today):
    g, cfg, events = ask_for_big_refund(monkeypatch)
    rid = events[-1]["payload"]["refund_id"]
    out = resolve_review(g, rid, approved=False, note="item was used")
    assert rt.get_refund(rid)["status"] == "rejected"
    assert "not approved" in out["message"] and "item was used" in out["message"]


def test_second_reviewer_gets_already_decided_and_nothing_changes(monkeypatch, frozen_today):
    g, cfg, events = ask_for_big_refund(monkeypatch)
    rid = events[-1]["payload"]["refund_id"]
    resolve_review(g, rid, approved=True)
    before = history(g, cfg)
    assert resolve_review(g, rid, approved=False, note="too late") == {"status": "already_decided"}
    assert rt.get_refund(rid)["status"] == "completed" and history(g, cfg) == before


def test_unknown_refund_id_is_already_decided(monkeypatch, frozen_today):
    g, _, _ = ask_for_big_refund(monkeypatch)
    assert resolve_review(g, 9999, approved=True) == {"status": "already_decided"}


def test_history_shows_text_messages_only_in_order(monkeypatch, frozen_today):
    g, cfg, _ = ask_for_big_refund(monkeypatch)
    h = history(g, cfg)
    assert h[0] == {"role": "user", "content": "Refund order 1011, it is damaged"}
    assert all(m["content"] for m in h) and {m["role"] for m in h} == {"user", "assistant"}  # tool calls skipped


def test_history_of_an_unknown_thread_is_empty():
    assert history(build_graph(), web_config({"id": 5, "name": "Eve", "email": "eve@example.com"})) == []


def test_pii_notice_is_reported_as_an_event(monkeypatch):
    stub_models(monkeypatch)
    monkeypatch.setattr(supervisor, "decide", lambda m: RouteDecision(next_agent="FINISH", reply="Hello!"))
    raw = {"messages": [HumanMessage("my card is 4111 1111 1111 1111")], "iterations": 0}  # skips new_message()
    events = list(turn_events(build_graph(), raw, web_config(ALICE)))
    assert any(e["type"] == "notice" for e in events)
    assert events[-1] == {"type": "reply", "node": "supervisor", "text": "Hello!"}
