"""Refund tools. Used ONLY by the Refund agent (the only agent with a write tool).

Policy is enforced in code, not by the LLM: `initiate_refund` re-runs the
eligibility check itself, so a model that skips the check (or is prompt-injected)
still cannot issue an ineligible refund.
"""
import sqlite3
from datetime import date, datetime
from pathlib import Path

from langchain_core.tools import tool

from ecommerce_agents import db

WINDOW_DAYS = 30
APPROVAL_THRESHOLD = 50.0  # refunds above this are queued for human approval

# refunds.status values
COMPLETED = "completed"
PENDING = "pending_approval"
REJECTED = "rejected"  # final: a rejected order cannot be re-requested
POLICY_PATH = Path(db.__file__).resolve().parents[2] / "data" / "return_policy.md"


def _order(conn, order_id: int):
    return conn.execute(
        "SELECT id, status, total, estimated_delivery FROM orders WHERE id = ?", (order_id,)
    ).fetchone()


def _refund_for(conn, order_id: int):
    return conn.execute("SELECT * FROM refunds WHERE order_id = ?", (order_id,)).fetchone()


def _status_problem(order) -> str | None:
    if order["status"] != "delivered":
        return f"Order status is '{order['status']}'; only delivered orders can be refunded."
    return None


def _window_problem(order, today: date) -> str | None:
    if not order["estimated_delivery"]:
        return "No delivery date on record."
    delivered = datetime.strptime(order["estimated_delivery"], "%Y-%m-%d").date()
    days = (today - delivered).days
    if days > WINDOW_DAYS:
        return f"Delivered {days} days ago; the return window is {WINDOW_DAYS} days."
    return None


def _eligibility(conn, order_id: int, today: date) -> dict:
    order = _order(conn, order_id)
    if order is None:
        return {"eligible": False, "reason": f"Order {order_id} not found."}
    base = {"order_id": order_id, "amount": order["total"]}
    problem = _status_problem(order)
    if problem:
        return {**base, "eligible": False, "reason": problem}
    existing = _refund_for(conn, order_id)
    if existing is not None:
        reasons = {
            PENDING: "A refund for this order is already awaiting human approval.",
            REJECTED: "A refund request for this order was reviewed and rejected; that decision is final.",
        }
        return {**base, "eligible": False,
                "reason": reasons.get(existing["status"], "This order has already been refunded.")}
    problem = _window_problem(order, today)
    if problem:
        return {**base, "eligible": False, "reason": problem}
    delivered = datetime.strptime(order["estimated_delivery"], "%Y-%m-%d").date()
    days = (today - delivered).days
    return {**base, "eligible": True, "reason": f"Delivered {days} days ago, within the {WINDOW_DAYS}-day window.",
            "requires_approval": order["total"] > APPROVAL_THRESHOLD}


def check_eligibility(order_id: int, today: date | None = None) -> dict:
    db.ensure_db()
    conn = db.get_conn()
    try:
        return _eligibility(conn, order_id, today or date.today())
    finally:
        conn.close()


def _existing_result(row) -> dict:
    """Result for an order that already has a refund row (any status)."""
    refund = dict(row)
    if row["status"] == COMPLETED:
        return {"success": True, "already_existed": True, "refund": refund}
    if row["status"] == PENDING:
        return {"success": False, "needs_human_approval": True, "already_existed": True,
                "refund": refund, "error": "A refund for this order is already awaiting human approval."}
    return {"success": False, "already_existed": True, "refund": refund,
            "error": "A refund request for this order was reviewed and rejected; that decision is final."}


def create_refund(order_id: int, reason: str, today: date | None = None) -> dict:
    """Issue a full refund if (and only if) the order passes the policy check.

    Amounts over APPROVAL_THRESHOLD are not paid out: a `pending_approval` row is
    created for a human to decide (see the approval node).
    """
    db.ensure_db()
    today = today or date.today()
    conn = db.get_conn()
    try:
        existing = _refund_for(conn, order_id)
        if existing is not None:  # idempotent: repeat calls never create a second refund
            return _existing_result(existing)
        check = _eligibility(conn, order_id, today)
        if not check["eligible"]:
            return {"success": False, "error": check["reason"]}
        needs_approval = check["requires_approval"]
        status = PENDING if needs_approval else COMPLETED
        try:
            with conn:
                cur = conn.execute(
                    "INSERT INTO refunds (order_id, amount, reason, status, created_at) VALUES (?,?,?,?,?)",
                    (order_id, check["amount"], reason, status, today.isoformat()),
                )
        except sqlite3.IntegrityError:  # lost a race with another request
            return _existing_result(_refund_for(conn, order_id))
        refund = dict(conn.execute("SELECT * FROM refunds WHERE id = ?", (cur.lastrowid,)).fetchone())
        if needs_approval:
            return {"success": False, "needs_human_approval": True, "already_existed": False,
                    "refund": refund,
                    "error": f"Refunds over ${APPROVAL_THRESHOLD:.0f} need human approval; "
                             "this request has been queued for a human agent."}
        return {"success": True, "already_existed": False, "refund": refund}
    finally:
        conn.close()


def get_refund(refund_id: int) -> dict | None:
    """Read one refund row (used by the approval node to show the reviewer)."""
    db.ensure_db()
    conn = db.get_conn()
    try:
        row = conn.execute("SELECT * FROM refunds WHERE id = ?", (refund_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def decide_refund(refund_id: int, approved: bool, note: str = "", today: date | None = None) -> dict:
    """Apply a human reviewer's decision to a pending refund. NOT an agent tool.

    Idempotent: only a `pending_approval` row can be decided, so a replayed node
    (LangGraph re-runs a node from the top on resume) never changes a decided refund.
    Approval re-checks status and window, because the request may have waited a while.
    """
    db.ensure_db()
    today = today or date.today()
    conn = db.get_conn()
    try:
        row = conn.execute("SELECT * FROM refunds WHERE id = ?", (refund_id,)).fetchone()
        if row is None:
            return {"success": False, "error": f"Refund {refund_id} not found."}
        if row["status"] != PENDING:
            return {"success": False, "already_decided": True, "refund": dict(row),
                    "error": f"Refund {refund_id} is already {row['status']}."}
        status, final_note = (COMPLETED if approved else REJECTED), note
        if approved:
            order = _order(conn, row["order_id"])
            problem = _status_problem(order) or _window_problem(order, today)
            if problem:
                status, final_note = REJECTED, f"Auto-rejected at approval time: {problem}"
        with conn:
            conn.execute("UPDATE refunds SET status = ?, reviewer_note = ? WHERE id = ?",
                         (status, final_note, refund_id))
        updated = dict(conn.execute("SELECT * FROM refunds WHERE id = ?", (refund_id,)).fetchone())
        return {"success": status == COMPLETED, "already_decided": False, "refund": updated}
    finally:
        conn.close()


@tool
def get_return_policy() -> str:
    """Get the store's return and refund policy text."""
    return POLICY_PATH.read_text(encoding="utf-8")


@tool
def check_return_eligibility(order_id: int) -> dict:
    """Check whether an order can be refunded under the return policy. Always call this before initiate_refund."""
    return check_eligibility(order_id)


@tool
def initiate_refund(order_id: int, reason: str) -> dict:
    """Issue a full refund for an order. Re-checks policy itself; refuses ineligible orders and queues refunds over $50 for human approval."""
    return create_refund(order_id, reason)


REFUND_TOOLS = [get_return_policy, check_return_eligibility, initiate_refund]
