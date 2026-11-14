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
