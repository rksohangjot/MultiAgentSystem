"""Shared by the Streamlit pages: one cached graph over the durable checkpointer."""
import streamlit as st

from ecommerce_agents import db
from ecommerce_agents.graph import build_graph, sqlite_checkpointer


@st.cache_resource
def get_graph():
    """Built once per server process (Streamlit re-runs the script on every click, the graph must not be
    rebuilt). The approval pause lives in this checkpointer, not in session state, so it survives reruns
    and a restart, and the reviewer page sees the same conversations as the customer page."""
    db.ensure_db()
    return build_graph(sqlite_checkpointer(db.DB_PATH.parent / "checkpoints.db"))


def demo_customers() -> list[dict]:
    """The mock accounts offered on the sign-in screen (stand-in for real authentication)."""
    db.ensure_db()
    conn = db.get_conn()
    try:
        return [dict(r) for r in conn.execute("SELECT id, name, email FROM customers ORDER BY id")]
    finally:
        conn.close()


def md(text: str) -> str:
    """Escape `$` so amounts like $89.00 are not rendered as LaTeX."""
    return text.replace("$", "\$")
