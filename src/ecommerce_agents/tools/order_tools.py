"""Read-only order tools. Used by the Order-Status agent."""
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from ecommerce_agents import db, identity


def lookup_order(order_id: int) -> dict:
    db.ensure_db()
    conn = db.get_conn()
    try:
        row = conn.execute(
            """SELECT o.id, o.status, o.tracking_number, o.order_date,
                      o.estimated_delivery, o.total, p.name AS product, c.name AS customer
               FROM orders o JOIN products p ON p.id = o.product_id
               JOIN customers c ON c.id = o.customer_id WHERE o.id = ?""",
            (order_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"error": f"Order {order_id} not found."}
    return dict(row)


@tool
def get_order_status(order_id: int, config: RunnableConfig) -> dict:
    """Get status, tracking number, product and estimated delivery for an order ID."""
    if not identity.owns_order(order_id, identity.current_customer_id(config)):
        return {"error": identity.not_found(order_id)}
    return lookup_order(order_id)


ORDER_TOOLS = [get_order_status]
