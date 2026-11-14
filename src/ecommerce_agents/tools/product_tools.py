"""Read-only product tools. Used by the Product-Info agent."""
from langchain_core.tools import tool

from ecommerce_agents import db


def find_products(query: str) -> list[dict]:
    db.ensure_db()
    like = f"%{query.strip()}%"
    conn = db.get_conn()
    try:
        rows = conn.execute(
            """SELECT id, name, category, price, stock, description FROM products
               WHERE name LIKE ? OR category LIKE ? OR description LIKE ? LIMIT 5""",
            (like, like, like),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


@tool
def search_products(query: str) -> list[dict] | dict:
    """Search the catalog by keyword (name, category or description). Returns price, stock and description."""
    results = find_products(query)
    return results or {"error": f"No products matched '{query}'."}


PRODUCT_TOOLS = [search_products]
