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


def test_refund_over_50_needs_human_approval():
    r = rt.create_refund(1011, "no longer needed", TODAY)
    assert not r["success"] and r["needs_human_approval"]
    assert refund_count() == 1


# --- tool wrappers ---
def test_policy_tool_returns_text():
    assert "30 days" in rt.get_return_policy.invoke({})


def test_refund_tools_exposed():
    assert [t.name for t in rt.REFUND_TOOLS] == ["get_return_policy", "check_return_eligibility", "initiate_refund"]
