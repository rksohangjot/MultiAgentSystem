"""SQLite access + mock data seeding."""
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("ECOM_DB_PATH", Path(__file__).resolve().parents[2] / "data" / "shop.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL,
    stock INTEGER, description TEXT);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER,
    status TEXT, tracking_number TEXT, order_date TEXT, estimated_delivery TEXT,
    total REAL);
CREATE TABLE IF NOT EXISTS refunds (
    id INTEGER PRIMARY KEY, order_id INTEGER UNIQUE, amount REAL,
    reason TEXT, status TEXT, created_at TEXT, reviewer_note TEXT);
"""

CUSTOMERS = [
    (1, "Alice Johnson", "alice@example.com"),
    (2, "Bob Smith", "bob@example.com"),
    (3, "Carol Lee", "carol@example.com"),
    (4, "Dan Brown", "dan@example.com"),
    (5, "Eve Davis", "eve@example.com"),
]

PRODUCTS = [
    (1, "Aero Wireless Headphones", "audio", 79.99, 25, "Bluetooth 5.3 over-ear headphones, 30h battery, noise cancelling."),
    (2, "TrailShield Rain Jacket", "apparel", 120.00, 8, "Waterproof breathable jacket, sizes S-XL, available in blue and black."),
    (3, "PixelPro 27in Monitor", "electronics", 249.00, 0, "27-inch 4K IPS monitor, USB-C 65W charging. Currently out of stock."),
    (4, "SparkLite Phone Case", "accessories", 19.99, 150, "Shockproof case for PhoneX 14, black or clear."),
    (5, "BrewMaster Coffee Maker", "home", 59.50, 12, "12-cup programmable coffee maker with thermal carafe."),
    (6, "Summit 40L Backpack", "outdoor", 89.00, 30, "Water-resistant hiking backpack with laptop sleeve."),
    (7, "ChargeMax 20W Charger", "accessories", 24.99, 80, "USB-C fast charger, compatible with PhoneX and most tablets."),
    (8, "Zen Yoga Mat", "fitness", 35.00, 40, "6mm non-slip yoga mat, eco-friendly TPE."),
    (9, "KeyForge Mechanical Keyboard", "electronics", 109.00, 5, "Hot-swappable 75% mechanical keyboard with RGB."),
    (10, "LumaDesk LED Lamp", "home", 42.00, 18, "Dimmable desk lamp with wireless phone charging base."),
]

ORDERS = [
    (1001, 1, 1, "delivered", "TRK1001", "2026-10-01", "2026-10-05", 79.99),
    (1002, 1, 4, "shipped", "TRK1002", "2026-11-08", "2026-11-14", 19.99),
    (1003, 2, 2, "processing", None, "2026-11-12", "2026-11-19", 120.00),
    (1004, 2, 5, "delayed", "TRK1004", "2026-10-28", "2026-11-05", 59.50),
    (1005, 3, 3, "cancelled", None, "2026-10-15", None, 249.00),
    (1006, 3, 9, "delivered", "TRK1006", "2026-09-20", "2026-09-25", 109.00),
    (1007, 4, 6, "shipped", "TRK1007", "2026-11-10", "2026-11-15", 89.00),
    (1008, 4, 7, "delivered", "TRK1008", "2026-11-01", "2026-11-04", 24.99),
    (1009, 5, 8, "processing", None, "2026-11-13", "2026-11-20", 35.00),
    (1010, 5, 10, "delivered", "TRK1010", "2026-10-10", "2026-10-14", 42.00),
    # Return-policy cases (relative to demo date 2026-11-13):
    # 1011: delivered recently, total > $50 (refund needs human approval)
    (1011, 1, 6, "delivered", "TRK1011", "2026-11-03", "2026-11-08", 89.00),
]

# 1010 is already refunded -> not eligible again. UNIQUE(order_id) blocks duplicates.
REFUNDS = [
    (1, 1010, 42.00, "Lamp stopped working", "completed", "2026-10-20"),
]


def get_conn(path: Path | str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def seed(path: Path | str | None = None) -> None:
    """(Re)create the database with mock data."""
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = get_conn(p)
    with conn:
        conn.executescript(
            "DROP TABLE IF EXISTS refunds; DROP TABLE IF EXISTS orders; DROP TABLE IF EXISTS products; DROP TABLE IF EXISTS customers;"
        )
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO customers VALUES (?,?,?)", CUSTOMERS)
        conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", PRODUCTS)
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", ORDERS)
        conn.executemany(
            "INSERT INTO refunds (id, order_id, amount, reason, status, created_at) VALUES (?,?,?,?,?,?)", REFUNDS)
    conn.close()


def ensure_db() -> None:
    if not DB_PATH.exists():
        seed()
        return
    # Light migration for databases created before the HITL feature.
    conn = get_conn()
    try:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(refunds)")]
        if "reviewer_note" not in cols:
            with conn:
                conn.execute("ALTER TABLE refunds ADD COLUMN reviewer_note TEXT")
    finally:
        conn.close()
