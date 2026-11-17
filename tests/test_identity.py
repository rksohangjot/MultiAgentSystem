"""Identity: tools only touch the signed-in customer's own orders (no API calls)."""
import pytest

from ecommerce_agents import db, identity
from ecommerce_agents.tools.order_tools import get_order_status
from ecommerce_agents.tools import refund_tools as rt

ME = {"configurable": {"customer_id": 4}}      # owns 1007, 1008
OTHER = {"configurable": {"customer_id": 1}}   # owns 1001, 1002, 1011


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


def test_find_customer_by_email_case_insensitive():
    assert identity.find_customer_by_email("  DAN@Example.com ")["id"] == 4
    assert identity.find_customer_by_email("nobody@example.com") is None


def test_own_order_is_visible():
    assert get_order_status.invoke({"order_id": 1008}, ME)["status"] == "delivered"


def test_other_customers_order_looks_missing():
    r = get_order_status.invoke({"order_id": 1001}, ME)
    assert r == {"error": "Order 1001 not found."}
    # identical to a genuinely missing order, so ids cannot be probed
    assert get_order_status.invoke({"order_id": 9999}, ME) == {"error": "Order 9999 not found."}


def test_no_session_fails_closed():
    assert "error" in get_order_status.invoke({"order_id": 1008}, {})
    assert not rt.check_return_eligibility.invoke({"order_id": 1008}, {})["eligible"]
    assert not rt.initiate_refund.invoke({"order_id": 1008, "reason": "x"}, {})["success"]
    assert refund_count() == 1


def test_cannot_check_or_refund_someone_elses_order():
    assert not rt.check_return_eligibility.invoke({"order_id": 1011}, ME)["eligible"]
    r = rt.initiate_refund.invoke({"order_id": 1011, "reason": "x"}, ME)
    assert r == {"success": False, "error": "Order 1011 not found."}
    assert refund_count() == 1  # nothing written


def test_owner_can_still_refund():
    assert rt.initiate_refund.invoke({"order_id": 1008, "reason": "x"}, ME)["success"]


def test_llm_cannot_supply_customer_id():
    for t in (get_order_status, rt.check_return_eligibility, rt.initiate_refund):
        assert "config" not in t.args and "customer_id" not in t.args


def test_sign_in_retries_then_succeeds_and_gives_up():
    from ecommerce_agents.chat_loop import sign_in
    answers = iter(["wrong@x.com", "bob@example.com"])
    assert sign_in(lambda _: next(answers), lambda _: None)["id"] == 2
    assert sign_in(lambda _: "nope@x.com", lambda _: None) is None


def test_session_config_is_per_customer():
    from ecommerce_agents.chat_loop import session_config
    a = session_config({"id": 1, "name": "A"})["configurable"]
    b = session_config({"id": 2, "name": "B"})["configurable"]
    assert a["customer_id"] == 1 and a["thread_id"] != b["thread_id"]
