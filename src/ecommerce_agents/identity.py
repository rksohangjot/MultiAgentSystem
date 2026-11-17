"""Who is the customer? Ownership checks for the tools.

The customer id comes from the session layer (the CLI today, real auth later) via
`config["configurable"]["customer_id"]`. It never passes through the LLM: tools read it
from the run config, so a prompt-injected model cannot choose a different customer.
Fail closed: no customer id means no access.
"""
from ecommerce_agents import db


def find_customer_by_email(email: str) -> dict | None:
    """Stand-in for real login until the Streamlit UI: look a customer up by email."""
    db.ensure_db()
    conn = db.get_conn()
    try:
        row = conn.execute("SELECT id, name, email FROM customers WHERE lower(email) = ?",
                           (email.strip().lower(),)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def current_customer_id(config) -> int | None:
    return ((config or {}).get("configurable") or {}).get("customer_id")


def owns_order(order_id: int, customer_id: int | None) -> bool:
    """True only if the order exists and belongs to this customer (False if not signed in)."""
    if customer_id is None:
        return False
    db.ensure_db()
    conn = db.get_conn()
    try:
        row = conn.execute("SELECT 1 FROM orders WHERE id = ? AND customer_id = ?",
                           (order_id, customer_id)).fetchone()
        return row is not None
    finally:
        conn.close()


def not_found(order_id: int) -> str:
    """One message for 'missing' and 'not yours', so order ids cannot be probed."""
    return f"Order {order_id} not found."
