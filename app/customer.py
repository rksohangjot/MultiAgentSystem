"""Customer chat page: sign in, talk to the support team, see refunds waiting for a human."""
import streamlit as st

from common import demo_customers, get_graph, md
from ecommerce_agents.chat_loop import new_message
from ecommerce_agents.web import has_pending, history, turn_events, web_config

EXAMPLES = ["Where is my order?", "Do you have wireless headphones?", "I'd like a refund for order 1011"]


def sign_in_view():
    st.title("Shop Support")
    st.caption("Demo accounts. Picking one stands in for real authentication.")
    customers = demo_customers()
    choice = st.selectbox("Sign in as", customers, format_func=lambda c: f"{c['name']} ({c['email']})")
    if st.button("Sign in", type="primary"):
        st.session_state["customer"] = choice
        st.rerun()


@st.fragment(run_every=3)
def pending_banner(graph, cfg):
    """Polls while a refund waits for a reviewer; when it is decided, re-run the page to show the outcome."""
    if not has_pending(graph, cfg):
        st.rerun()
    st.info("Your refund request is waiting for a human reviewer. This page updates when they decide.",
            icon="⏳")


def handle_turn(graph, cfg, customer, text):
    with st.chat_message("user"):
        st.markdown(md(text))
    pending, route = False, []
    with st.chat_message("assistant"):
        status = st.status("Working on it...", expanded=False)
        for event in turn_events(graph, new_message(text, customer), cfg):
            if event["type"] == "notice":
                st.warning(event["text"], icon="🔒")
            elif event["type"] == "route" and event["agent"] != "FINISH":
                route.append(event["agent"])
                status.update(label="Routed to: " + " → ".join(route))
            elif event["type"] == "reply":
                st.markdown(md(event["text"]))
            elif event["type"] == "pending":
                pending = True
        status.update(label=("Routed to: " + " → ".join(route)) if route else "Answered directly",
                      state="complete")
    if pending:
        st.rerun()  # re-render from the checkpoint so the banner shows and the input locks


def render():
    customer = st.session_state.get("customer")
    if not customer:
        return sign_in_view()
    graph, cfg = get_graph(), web_config(customer)

    with st.sidebar:
        st.write(f"Signed in as **{customer['name']}**")
        if st.button("Sign out"):
            del st.session_state["customer"]
            st.rerun()
        st.caption("Try:")
        for example in EXAMPLES:
            st.caption(f"• {example}")

    st.title("Shop Support")
    for message in history(graph, cfg):
        with st.chat_message(message["role"]):
            st.markdown(md(message["content"]))

    pending = has_pending(graph, cfg)
    if pending:
        pending_banner(graph, cfg)
    text = st.chat_input("Ask about an order, a product, or a refund", disabled=pending)
    if text:
        handle_turn(graph, cfg, customer, text)


render()
