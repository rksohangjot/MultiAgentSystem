"""The dataset must be well-formed so a typo cannot become a silent bad eval. Offline, no LLM."""
from collections import Counter

import pytest

from ecommerce_agents import db
from ecommerce_agents.evals.checks import CHECKERS
from ecommerce_agents.evals.dataset import ALL_TOOLS, CASES
from ecommerce_agents.tools import AGENT_TOOLS

CATEGORIES = {"routing", "refund_policy", "adversarial", "mixed", "permissions"}
CUSTOMER_IDS = {c[0] for c in db.CUSTOMERS}
ORDER_IDS = {o[0] for o in db.ORDERS}
TOOL_NAMES = {t.name for tools in AGENT_TOOLS.values() for t in tools}


def test_ids_are_unique():
    dupes = [i for i, n in Counter(c.id for c in CASES).items() if n > 1]
    assert not dupes


def test_size_and_category_mix():
    assert 30 <= len(CASES) <= 40
    counts = Counter(c.category for c in CASES)
    assert set(counts) == CATEGORIES
    assert all(n >= 4 for n in counts.values())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_case_is_well_formed(case):
    assert case.category in CATEGORIES
    assert case.customer_id in CUSTOMER_IDS
    assert case.messages and all(m.strip() for m in case.messages)
    assert set(case.expect) <= set(CHECKERS), f"unknown expect keys: {set(case.expect) - set(CHECKERS)}"
    assert case.expect, "every case needs at least one expectation"
    for oid in case.expect.get("refund_state", {}):
        assert oid in ORDER_IDS
    for name in case.expect.get("tools_called", []) + case.expect.get("tools_not_called", []):
        assert name in TOOL_NAMES
    # A scripted reviewer decision must exist for every approval pause the case expects.
    assert len(case.reviewer) >= case.expect.get("reviews", 0)


def test_all_tools_constant_matches_real_tools():
    assert set(ALL_TOOLS) == TOOL_NAMES


def test_judge_is_used_sparingly():
    judged = sum(c.judge for c in CASES)
    assert 5 <= judged <= 15
