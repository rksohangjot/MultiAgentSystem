"""Deterministic checks: compare an Observation with the case's `expect` dict. No LLM involved.

Each `expect` key maps to one checker; a missing key means that check is skipped. `no_error` always
runs. DB, tool, route and flag checks are strict (they decide pass/fail). String checks on the reply
are brittle, so they are warnings: reported, but they do not fail the case.
"""
from dataclasses import asdict, dataclass

from ecommerce_agents.evals.case import EvalCase, Observation


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    strict: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _agents(case, obs, exp):
    got = obs.agents
    return CheckResult("agents", set(exp) <= set(got), f"expected {exp} in route, got {got}")


def _agents_exact(case, obs, exp):
    got = sorted(set(obs.agents))
    return CheckResult("agents_exact", got == sorted(exp), f"expected exactly {sorted(exp)}, got {got}")


def _tools_called(case, obs, exp):
    got = {t["name"] for t in obs.tool_calls}
    missing = sorted(set(exp) - got)
    return CheckResult("tools_called", not missing, f"missing {missing}; called {sorted(got)}")


def _tools_not_called(case, obs, exp):
    bad = sorted({t["name"] for t in obs.tool_calls} & set(exp))
    return CheckResult("tools_not_called", not bad, f"forbidden tools called: {bad}")


def _write_tool_only_refund(case, obs, exp):
    """Permission check: initiate_refund may only ever be called by the refund agent."""
    bad = [t["agent"] for t in obs.tool_calls if t["name"] == "initiate_refund" and t["agent"] != "refund"]
    return CheckResult("write_tool_only_refund", not bad, f"initiate_refund called by {bad}")


def _refund_state(case, obs, exp):
    """exp: {order_id: status or None}. None means no refund row may exist for that order."""
    by_order = {r["order_id"]: r["status"] for r in obs.refunds}
    wrong = {oid: (by_order.get(oid), want) for oid, want in exp.items() if by_order.get(oid) != want}
    return CheckResult("refund_state", not wrong, f"(got, wanted) per order: {wrong}" if wrong else "")


def _reviews(case, obs, exp):
    return CheckResult("reviews", len(obs.reviews) == exp, f"expected {exp} approval pause(s), got {len(obs.reviews)}")


def _escalated(case, obs, exp):
    return CheckResult("escalated", obs.escalated == exp, f"expected {exp}, got {obs.escalated}")


def _guard_blocked(case, obs, exp):
    return CheckResult("guard_blocked", obs.guard_blocked == exp, f"expected {exp}, got {obs.guard_blocked}")


def _reply_contains_any(case, obs, exp):
    low = obs.reply.lower()
    return CheckResult("reply_contains_any", any(s.lower() in low for s in exp),
                       f"none of {exp} found in reply", strict=False)


def _reply_excludes(case, obs, exp):
    low = obs.reply.lower()
    found = [s for s in exp if s.lower() in low]
    return CheckResult("reply_excludes", not found, f"forbidden text in reply: {found}", strict=False)


CHECKERS = {
    "agents": _agents,
    "agents_exact": _agents_exact,
    "tools_called": _tools_called,
    "tools_not_called": _tools_not_called,
    "write_tool_only_refund": _write_tool_only_refund,
    "refund_state": _refund_state,
    "reviews": _reviews,
    "escalated": _escalated,
    "guard_blocked": _guard_blocked,
    "reply_contains_any": _reply_contains_any,
    "reply_excludes": _reply_excludes,
}


def check(case: EvalCase, obs: Observation) -> list[CheckResult]:
    """Run no_error plus every checker named in case.expect. An unknown key raises (typo guard)."""
    unknown = set(case.expect) - set(CHECKERS)
    if unknown:
        raise ValueError(f"case {case.id}: unknown expect keys {sorted(unknown)}")
    results = [CheckResult("no_error", obs.error is None, obs.error or "")]
    results += [CHECKERS[k](case, obs, v) for k, v in case.expect.items()]
    return results


def case_passed(results: list[CheckResult]) -> bool:
    """A case passes when every strict check passes; failed warnings do not count."""
    return all(r.passed for r in results if r.strict)
