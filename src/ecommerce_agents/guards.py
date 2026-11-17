"""Input guard, pure functions: prompt-injection heuristics and PII redaction.

Deliberately simple and deterministic (no LLM). It is a first line of defence, not the
only one: tools are permissioned per customer and per agent, and refund policy is code.
"""
import re
from dataclasses import dataclass, field

# --- prompt injection ------------------------------------------------------
_INJECTION = [
    ("override-instructions", r"\b(ignore|disregard|forget|override)\b[^.\n]{0,30}\b(previous|prior|above|earlier|all|any|your|the)\b[^.\n]{0,20}\b(instructions?|rules?|prompts?|guidelines?)\b"),
    ("persona-switch", r"\b(you are now|from now on you are|pretend (to be|you are)|act as (an? )?(unrestricted|admin|developer|dan)\b)"),
    ("prompt-extraction", r"\b(reveal|show|print|repeat|tell me|output)\b[^.\n]{0,15}\b(your|the)\s+(system|initial|hidden|original)\s+(prompt|instructions?)\b"),
    ("jailbreak", r"\b(jailbreak|developer mode|do anything now)\b"),
    ("force-approval", r"\b(approve|authori[sz]e|process)\b[^.\n]{0,12}\b(this|my|the|that)\s+refund\b[^.\n]{0,30}\b(without|regardless|no matter|override)"),
    ("policy-bypass", r"\b(bypass|override|skip|disable)\b[^.\n]{0,15}\b(the\s+)?(policy|approval|eligibility|human review|safety|limit)\b"),
    ("fake-role-marker", r"(^|\n)\s*(system|assistant|developer)\s*:|<\|im_start\|>|\[/?INST\]"),
]
_INJECTION_RE = [(name, re.compile(p, re.IGNORECASE)) for name, p in _INJECTION]

REFUSAL = ("I can't act on that request. I can help with order status, product questions, "
           "and returns/refunds. What would you like to do?")
REMOVED = "[message removed by safety filter]"


def detect_injection(text: str) -> str | None:
    """Name of the first matching injection pattern, or None."""
    for name, rx in _INJECTION_RE:
        if rx.search(text):
            return name
    return None


# --- PII -------------------------------------------------------------------
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_SSN = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w-])(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?![\w-])")


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


@dataclass
class PiiResult:
    text: str
    found: list[str] = field(default_factory=list)  # kinds redacted, in order of detection


def redact_pii(text: str, own_email: str | None = None) -> PiiResult:
    """Replace card numbers, SSNs, phone numbers and other people's emails with tags.

    The customer's own email is left alone (it is how they identify themselves).
    """
    found: list[str] = []

    def card(m):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            found.append("card")
            return "[REDACTED_CARD]"
        return m.group()

    def email(m):
        if own_email and m.group().lower() == own_email.lower():
            return m.group()
        found.append("email")
        return "[REDACTED_EMAIL]"

    def simple(kind, tag):
        def sub(m):
            found.append(kind)
            return tag
        return sub

    text = _CARD.sub(card, text)
    text = _SSN.sub(simple("ssn", "[REDACTED_SSN]"), text)
    text = _EMAIL.sub(email, text)
    text = _PHONE.sub(simple("phone", "[REDACTED_PHONE]"), text)
    return PiiResult(text, found)


def pii_notice(found: list[str]) -> str:
    kinds = ", ".join(sorted(set(found)))
    return (f"For your security I removed personal data ({kinds}) from your message. "
            "I never need it; please don't share it in chat.")
