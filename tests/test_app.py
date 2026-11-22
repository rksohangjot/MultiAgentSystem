"""Streamlit pages, run headless with Streamlit's AppTest against the stubbed graph (no API calls).

The pages import `common` from the app folder; here `common.get_graph` is replaced by a graph built
with stubbed models, so the customer page and the reviewer page share one checkpointer like in production.
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import common  # noqa: E402

from ecommerce_agents.chat_loop import new_message  # noqa: E402
from ecommerce_agents.graph import build_graph  # noqa: E402
from ecommerce_agents.tools import refund_tools as rt  # noqa: E402
from ecommerce_agents.web import history, turn_events, web_config  # noqa: E402
from tests.test_refund_graph import big_refund_model, frozen_today, seeded, stub_models, stub_supervisor  # noqa: E402,F401

ALICE = {"id": 1, "name": "Alice Johnson", "email": "alice@example.com"}
PASSWORD = "s3cret-pass"


@pytest.fixture
def shared(monkeypatch):
    """One graph (stubbed refund model for order 1011) shared by every AppTest in the test."""
    stub_supervisor(monkeypatch, ["refund", "FINISH", "FINISH", "FINISH"])
    stub_models(monkeypatch, refund_model=big_refund_model())
    graph = build_graph()
    monkeypatch.setattr(common, "get_graph", lambda: graph)
    monkeypatch.setenv("REVIEWER_PASSWORD", PASSWORD)
    return graph


def page(name, **session):
    at = AppTest.from_file(str(APP / name), default_timeout=30)
    for key, value in session.items():
        at.session_state[key] = value
    return at.run()


def queue_refund(graph):
    """Alice asks for the $89 refund through the logic layer; returns the refund id waiting for review."""
    events = list(turn_events(graph, new_message("Refund order 1011, it is damaged", ALICE), web_config(ALICE)))
    return events[-1]["payload"]["refund_id"]


def texts(elements):
    return [e.value for e in elements]


# --- reviewer page ---------------------------------------------------------------------------

def test_reviewer_locked_when_no_password_is_configured(shared, monkeypatch):
    monkeypatch.delenv("REVIEWER_PASSWORD")
    at = page("reviewer.py")
    assert "not configured" in at.error[0].value and len(at.text_input) == 0  # fails closed


def test_wrong_passphrase_is_rejected_and_shows_no_queue(shared, frozen_today):
    queue_refund(shared)
    at = page("reviewer.py")
    at.text_input[0].set_value("guess").run()
    at.button[0].click().run()
    assert any("Wrong passphrase" in e.value for e in at.error)
    assert not any("Refund #" in m.value for m in at.markdown)


def test_correct_passphrase_shows_an_empty_queue(shared):
    at = page("reviewer.py", reviewer_ok=True)
    assert any("No refunds are waiting" in i.value for i in at.info)


def test_queue_shows_the_pending_refund_with_its_details(shared, frozen_today):
    rid = queue_refund(shared)
    at = page("reviewer.py", reviewer_ok=True)
    body = " ".join(texts(at.markdown))
    assert f"Refund #{rid}" in body and "order 1011" in body and "89.00" in body and "Alice Johnson" in body


def test_approve_pays_the_refund_and_tells_the_customer(shared, frozen_today):
    rid = queue_refund(shared)
    at = page("reviewer.py", reviewer_ok=True)
    at.button(key=f"ok-{rid}").click().run()
    assert rt.get_refund(rid)["status"] == "completed"
    assert "approved and processed" in history(shared, web_config(ALICE))[-1]["content"]
    assert any("Approved" in t.value for t in at.toast)


def test_reject_without_a_reason_is_refused(shared, frozen_today):
    rid = queue_refund(shared)
    at = page("reviewer.py", reviewer_ok=True)
    at.button(key=f"no-{rid}").click().run()
    assert any("give a reason" in w.value for w in at.warning)
    assert rt.get_refund(rid)["status"] == "pending_approval"  # nothing decided


def test_reject_with_a_reason_is_final_and_shown_to_the_customer(shared, frozen_today):
    rid = queue_refund(shared)
    at = page("reviewer.py", reviewer_ok=True)
    at.text_input(key=f"note-{rid}").set_value("item was used").run()
    at.button(key=f"no-{rid}").click().run()
    assert rt.get_refund(rid)["status"] == "rejected"
    last = history(shared, web_config(ALICE))[-1]["content"]
    assert "not approved" in last and "item was used" in last


def test_a_click_on_a_refund_another_reviewer_already_decided_changes_nothing(shared, frozen_today):
    from ecommerce_agents.web import resolve_review
    rid = queue_refund(shared)
    at = page("reviewer.py", reviewer_ok=True)  # this reviewer still has the card open...
    resolve_review(shared, rid, approved=True)  # ...while another reviewer decides first
    at.button(key=f"no-{rid}").click().run()  # the card is gone on the re-run, so the click is ignored
    assert not at.exception and rt.get_refund(rid)["status"] == "completed"  # the first decision stands
    assert any("No refunds are waiting" in i.value for i in at.info)


# --- customer page ---------------------------------------------------------------------------

def test_sign_in_screen_offers_the_demo_accounts(shared):
    at = page("customer.py")
    assert len(at.selectbox[0].options) == 5 and at.button[0].label == "Sign in" and len(at.chat_input) == 0


def test_signing_in_opens_the_chat(shared):
    at = page("customer.py")
    at.button[0].click().run()
    assert at.session_state["customer"]["id"] == 1 and len(at.chat_input) == 1
    assert not at.exception


def test_refund_request_locks_the_input_and_shows_the_waiting_banner(shared, frozen_today):
    at = page("customer.py", customer=ALICE)
    at.chat_input[0].set_value("Refund order 1011, it is damaged").run()
    assert not at.exception
    assert at.chat_input[0].disabled  # no new messages while a human decides
    assert any("waiting for a human reviewer" in i.value for i in at.info)
    assert any("human agent will review" in m.value for m in at.chat_message[-1].markdown)


def test_after_approval_the_customer_sees_the_outcome_and_can_chat_again(shared, frozen_today):
    from ecommerce_agents.web import resolve_review
    at = page("customer.py", customer=ALICE)
    at.chat_input[0].set_value("Refund order 1011, it is damaged").run()
    resolve_review(shared, queue_pending_id(), approved=True)
    at.run()
    assert not any("waiting for a human reviewer" in i.value for i in at.info)
    assert not at.chat_input[0].disabled
    assert "approved and processed" in " ".join(texts(at.chat_message[-1].markdown))


def queue_pending_id():
    return rt.pending_refunds()[0]["refund_id"]


def test_dollar_amounts_are_escaped_so_they_do_not_render_as_math(shared, frozen_today):
    at = page("customer.py", customer=ALICE)
    at.chat_input[0].set_value("Refund order 1011, it is damaged").run()
    shown = " ".join(texts(at.chat_message[-1].markdown))
    assert "$" not in shown.replace("\\$", "")  # every $ is escaped


def test_signing_out_returns_to_the_sign_in_screen(shared):
    at = page("customer.py", customer=ALICE)
    at.sidebar.button[0].click().run()
    assert "customer" not in at.session_state and len(at.selectbox) == 1
