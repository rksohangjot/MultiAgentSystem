"""Reviewer page: the human in the loop. Lists refunds waiting for a decision and resumes the customer's thread.

Access is a shared passphrase from REVIEWER_PASSWORD. This is a demo gate, not real authentication, and
it fails closed: with no password configured nobody can open the queue.
"""
import hmac
import os

import streamlit as st

from common import get_graph, md
from ecommerce_agents.web import pending_reviews, resolve_review


def gate() -> bool:
    expected = os.getenv("REVIEWER_PASSWORD", "")
    if not expected:
        st.error("Reviewer access is not configured. Set REVIEWER_PASSWORD in .env and restart the app.")
        return False
    if st.session_state.get("reviewer_ok"):
        return True
    st.caption("Demo gate: a shared passphrase stands in for real reviewer authentication.")
    entered = st.text_input("Reviewer passphrase", type="password")
    if st.button("Enter", type="primary"):
        if hmac.compare_digest(entered.encode(), expected.encode()):
            st.session_state["reviewer_ok"] = True
            st.rerun()
        st.error("Wrong passphrase.")
    return False


def decide(graph, refund_id: int, approved: bool, note: str):
    if not approved and not note.strip():
        st.warning("Please give a reason: the customer is shown it.")
        return
    result = resolve_review(graph, refund_id, approved, note)
    if result["status"] == "already_decided":
        st.toast("Someone else already decided this refund.", icon="⚠️")
    else:
        st.toast(("Approved. " if approved else "Rejected. ") + "The customer has been told.", icon="✅")
    st.rerun()


@st.fragment(run_every=4)
def queue(graph):
    items = pending_reviews(graph)
    if not items:
        st.info("No refunds are waiting for review.", icon="✅")
        return
    st.write(f"**{len(items)}** waiting for review")
    for item in items:
        with st.container(border=True):
            amount = md(f"${item['amount']:.2f}")
            st.markdown(f"**Refund #{item['refund_id']}**: order {item['order_id']}, "
                        f"**{amount}**, customer {item['customer_name']}")
            st.caption(f"Reason given: {item['reason']!r} · requested {item['created_at']}")
            note = st.text_input("Note (required when rejecting; shown to the customer)",
                                 key=f"note-{item['refund_id']}")
            approve, reject = st.columns(2)
            if approve.button("Approve", key=f"ok-{item['refund_id']}", type="primary"):
                decide(graph, item["refund_id"], True, note)
            if reject.button("Reject", key=f"no-{item['refund_id']}"):
                decide(graph, item["refund_id"], False, note)


def render():
    st.title("Refund review")
    if not gate():
        return
    if st.sidebar.button("Leave reviewer mode"):
        del st.session_state["reviewer_ok"]
        st.rerun()
    queue(get_graph())


render()
