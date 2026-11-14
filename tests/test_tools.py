import pytest

from ecommerce_agents import db
from ecommerce_agents.tools.order_tools import lookup_order
from ecommerce_agents.tools.product_tools import find_products


@pytest.fixture(autouse=True)
def seeded(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.seed(path)


def test_order_found():
    o = lookup_order(1004)
    assert o["status"] == "delayed" and o["product"] == "BrewMaster Coffee Maker"


def test_order_missing():
    assert "error" in lookup_order(9999)


def test_product_search_and_stock():
    r = find_products("monitor")
    assert r and r[0]["stock"] == 0


def test_product_no_match():
    assert find_products("zzzz") == []


def test_refunds_table_seeded():
    conn = db.get_conn()
    try:
        rows = conn.execute("SELECT order_id, status FROM refunds").fetchall()
    finally:
        conn.close()
    assert [(r["order_id"], r["status"]) for r in rows] == [(1010, "completed")]


def test_one_refund_per_order_enforced():
    import sqlite3
    conn = db.get_conn()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO refunds (order_id, amount) VALUES (1010, 1.0)")
    finally:
        conn.close()


def test_policy_file_exists():
    assert (db.DB_PATH.parent / "return_policy.md").exists() or \
        (db.Path(db.__file__).resolve().parents[2] / "data" / "return_policy.md").exists()
