"""Output guard: checks every reply before the customer sees it. Deterministic, no LLM.

Blocks replies that
- claim a refund was issued/approved when the database has no completed refund for it,
- leak internals (tool names, prompt text, state field names),
- expose other customers' data (their email or full name) or raw card/SSN/phone numbers.

A blocked reply is replaced (same message id) with a safe fallback that escalates to a
human. The guard wraps each node that produces customer-visible text (`guarded`), so the
replacement happens before the CLI/UI streams the node's output.
"""
import logging
import re

from ecommerce_agents import db, guards, identity, tracing

log = logging.getLogger(__name__)

FALLBACK = "Sorry, I can't give you a reliable answer to that. Let me get a human to help."

# --- unbacked refund claims ------------------------------------------------
_DONE = r"(?:issued|processed|approved|completed|sent|initiated|granted|paid)"
_CLAIMS = [re.compile(p, re.IGNORECASE) for p in (
    rf"\brefund\b[^.!?\n]{{0,60}}?\b(?:has|have|had|was|is|been|now)\b[^.!?\n]{{0,20}}?\b{_DONE}\b",
    rf"\b(?:i|we)(?:'ve| have)?\s+(?:just\s+)?{_DONE}\b[^.!?\n]{{0,30}}\brefund\b",
    r"\b(?:been|have|has|was|i've|we've)\s+refunded\b",
    r"\b(?:i|we)\s+(?:have\s+|just\s+)?refunded\b",
)]
# A sentence that hedges, denies or is conditional is not a claim that money moved.
_HEDGE = re.compile(r"\b(?:not|never|no|cannot|unable|if|once|until|unless|pending|awaiting|"
                    r"would|will|may|might|could|whether|when|queued)\b|n't|can't", re.IGNORECASE)
_ORDER_REF = re.compile(r"\border\s*(?:id|number|no\.?)?\s*#?\s*(\d{3,})", re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def _completed_refund_orders(customer_id: int | None) -> set[int]:
    if customer_id is None:
        return set()
    db.ensure_db()
    conn = db.get_conn()
    try:
        rows = conn.execute(
            "SELECT r.order_id FROM refunds r JOIN orders o ON o.id = r.order_id "
            "WHERE o.customer_id = ? AND r.status = 'completed'", (customer_id,)).fetchall()
        return {r["order_id"] for r in rows}
    finally:
        conn.close()


def unbacked_refund_claim(text: str, customer_id: int | None) -> bool:
    """True if the reply says a refund was paid/approved but the DB does not back it."""
    claim_sentences = [s for s in _SENTENCES.split(text)
                       if any(c.search(s) for c in _CLAIMS) and not _HEDGE.search(s)]
    if not claim_sentences:
        return False
    backed = _completed_refund_orders(customer_id)
    whole_ids = {int(i) for i in _ORDER_REF.findall(text)}
    for sentence in claim_sentences:
        ids = {int(i) for i in _ORDER_REF.findall(sentence)} or whole_ids
        if not (ids <= backed if ids else backed):
            return True
    return False


# --- leaks -----------------------------------------------------------------------
_INTERNALS = re.compile(
    r"\b(?:get_order_status|search_products|get_return_policy|check_return_eligibility|"
    r"initiate_refund|decide_refund|pending_refund_id|next_agent|reviewer_note|thread_id|customer_id)\b"
    r"|system prompt|\byou are the (?:order-status|returns/refund|product-info|supervisor)\b",
    re.IGNORECASE)


def leaks_internals(text: str) -> bool:
    return bool(_INTERNALS.search(text))


def exposes_other_customer(text: str, customer_id: int | None) -> bool:
    """Other customers' emails or full names, or raw card/SSN/phone numbers."""
    own = identity.get_customer(customer_id)
    if guards.redact_pii(text, own_email=own["email"] if own else None).found:
        return True
    db.ensure_db()
    conn = db.get_conn()
    try:
        others = conn.execute("SELECT name FROM customers WHERE id IS NOT ?", (customer_id,)).fetchall()
    finally:
        conn.close()
    low = text.lower()
    return any(r["name"].lower() in low for r in others)


def check_reply(text: str, customer_id: int | None) -> str | None:
    """Name of the first failed check, or None if the reply is fine."""
    if leaks_internals(text):
        return "leak"
    if exposes_other_customer(text, customer_id):
        return "other-customer-data"
    if unbacked_refund_claim(text, customer_id):
        return "unbacked-refund-claim"
    return None


# --- graph integration -----------------------------------------------------------
def guard_update(update: dict, state, config) -> dict:
    """Replace unsafe new AI replies in a node's update. Other messages pass through."""
    messages = update.get("messages")
    if not messages:
        return update
    seen = {m.id for m in state["messages"] if m.id}
    customer_id = identity.current_customer_id(config)
    out = []
    for m in messages:
        is_new_reply = (m.type == "ai" and (m.id is None or m.id not in seen)
                        and isinstance(m.content, str) and m.content.strip())
        reason = check_reply(m.content, customer_id) if is_new_reply else None
        if reason:
            log.warning("output guard blocked a reply: %s", reason)
            tracing.record_event("guard.output.blocked", reason=reason)
            m = m.model_copy(update={"content": FALLBACK, "tool_calls": [], "additional_kwargs": {}})
        out.append(m)
    return {**update, "messages": out}


def guarded(node):
    """Wrap a `(state, config) -> update` node so its replies are checked before streaming."""
    def wrapper(state, config):
        return guard_update(node(state, config), state, config)
    return wrapper
