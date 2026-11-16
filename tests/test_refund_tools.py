from datetime import date

import pytest

from ecommerce_agents import db
from ecommerce_agents.tools import refund_tools as rt

TODAY = date(2026, 11, 13)


@pytest.fixture(autouse=True)
def seeded(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.seed(path)


def refund_count():
    conn = db.get_conn()
    try:
        return conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0]
    finally:
        conn.close()


# --- eligibility ---
def test_eligible_recent_delivery():
    r = rt.check_eligibility(1008, TODAY)
    assert r["eligible"] and r["amount"] == 24.99 and r["requires_approval"] is False


def test_eligible_but_needs_approval_over_50():
    r = rt.check_eligibility(1011, TODAY)
    assert r["eligible"] and r["requires_approval"] is True


@pytest.mark.parametrize("order_id", [1001, 1006])
def test_outside_window(order_id):
    r = rt.check_eligibility(order_id, TODAY)
    assert not r["eligible"] and "return window" in r["reason"]


def test_already_refunded():
    r = rt.check_eligibility(1010, TODAY)
    assert not r["eligible"] and "already been refunded" in r["reason"]


@pytest.mark.parametrize("order_id", [1002, 1003, 1004, 1005])
def test_not_delivered(order_id):
    r = rt.check_eligibility(order_id, TODAY)
    assert not r["eligible"] and "delivered" in r["reason"]


def test_missing_order():
    r = rt.check_eligibility(9999, TODAY)
    assert not r["eligible"] and "not found" in r["reason"]


def test_window_boundary_day_30_ok_day_31_not():
    # order 1008 delivered 2026-11-04
    assert rt.check_eligibility(1008, date(2026, 12, 4))["eligible"]
    assert not rt.check_eligibility(1008, date(2026, 12, 5))["eligible"]


# --- initiate refund ---
def test_refund_success_writes_row():
    r = rt.create_refund(1008, "changed mind", TODAY)
    assert r["success"] and not r["already_existed"]
    assert r["refund"]["amount"] == 24.99 and r["refund"]["created_at"] == "2026-11-13"
    assert refund_count() == 2  # seeded 1010 + new


def test_refund_is_idempotent():
    first = rt.create_refund(1008, "x", TODAY)
    second = rt.create_refund(1008, "x", TODAY)
    assert second["already_existed"] and second["refund"]["id"] == first["refund"]["id"]
    assert refund_count() == 2


def test_refund_rejects_ineligible_without_prior_check():
    # The tool must enforce policy itself, even if the agent never called the check.
    r = rt.create_refund(1001, "late", TODAY)
    assert not r["success"] and "return window" in r["error"]
    assert refund_count() == 1


def test_refund_over_50_is_queued_not_paid():
    r = rt.create_refund(1011, "no longer needed", TODAY)
    assert not r["success"] and r["needs_human_approval"] and not r["already_existed"]
    assert r["refund"]["status"] == "pending_approval" and r["refund"]["amount"] == 89.00
    assert refund_count() == 2  # seeded 1010 + pending row


def test_pending_request_is_idempotent():
    first = rt.create_refund(1011, "x", TODAY)
    second = rt.create_refund(1011, "x", TODAY)
    assert second["needs_human_approval"] and second["already_existed"]
    assert second["refund"]["id"] == first["refund"]["id"]
    assert refund_count() == 2


def test_pending_order_not_eligible_again():
    rt.create_refund(1011, "x", TODAY)
    r = rt.check_eligibility(1011, TODAY)
    assert not r["eligible"] and "awaiting human approval" in r["reason"]


def test_rejected_order_is_final():
    rt.create_refund(1011, "x", TODAY)
    conn = db.get_conn()
    with conn:
        conn.execute("UPDATE refunds SET status = 'rejected' WHERE order_id = 1011")
    conn.close()
    assert "final" in rt.check_eligibility(1011, TODAY)["reason"]
    r = rt.create_refund(1011, "please", TODAY)
    assert not r["success"] and not r.get("needs_human_approval") and "final" in r["error"]
    assert refund_count() == 2


# --- tool wrappers ---
def test_policy_tool_returns_text():
    assert "30 days" in rt.get_return_policy.invoke({})


def test_refund_tools_exposed():
    assert [t.name for t in rt.REFUND_TOOLS] == ["get_return_policy", "check_return_eligibility", "initiate_refund"]


# --- human decision on a pending refund ---
def _pending(order_id=1011):
    return rt.create_refund(order_id, "changed mind", TODAY)["refund"]["id"]


def _row(refund_id):
    return rt.get_refund(refund_id)


def test_approve_completes_refund():
    rid = _pending()
    r = rt.decide_refund(rid, True, "looks fine", TODAY)
    assert r["success"] and _row(rid)["status"] == "completed" and _row(rid)["reviewer_note"] == "looks fine"


def test_reject_is_final_and_records_note():
    rid = _pending()
    r = rt.decide_refund(rid, False, "item was used", TODAY)
    assert not r["success"] and _row(rid)["status"] == "rejected"
    assert "final" in rt.check_eligibility(1011, TODAY)["reason"]


def test_decision_is_idempotent():
    rid = _pending()
    rt.decide_refund(rid, True, "", TODAY)
    again = rt.decide_refund(rid, False, "oops", TODAY)  # replay with a different answer
    assert again["already_decided"] and _row(rid)["status"] == "completed"


def test_approval_rechecks_window():
    rid = _pending()  # order 1011 delivered 2026-11-08
    r = rt.decide_refund(rid, True, "", date(2026, 12, 20))  # sat in the queue past 30 days
    assert not r["success"] and _row(rid)["status"] == "rejected"
    assert "Auto-rejected" in _row(rid)["reviewer_note"]


def test_decide_unknown_refund():
    assert not rt.decide_refund(999, True)["success"]


def test_decide_refund_is_not_an_agent_tool():
    from ecommerce_agents.tools import AGENT_TOOLS
    all_names = {t.name for tools in AGENT_TOOLS.values() for t in tools}
    assert "decide_refund" not in all_names
