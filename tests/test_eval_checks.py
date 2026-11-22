"""Checker unit tests on hand-built Observations: no graph, no API calls."""
import pytest

from ecommerce_agents.evals.case import EvalCase, Observation
from ecommerce_agents.evals.checks import case_passed, check


def run(expect, **obs_fields):
    case = EvalCase("c", "routing", 1, ["hi"], expect=expect)
    return check(case, Observation("c", **obs_fields))


def result(results, name):
    return next(r for r in results if r.name == name)


def tool(agent, name):
    return {"agent": agent, "name": name, "args": {}, "result": ""}


def test_no_error_always_runs_and_fails_on_error():
    assert case_passed(run({}))
    res = run({}, error="boom")
    assert not result(res, "no_error").passed and not case_passed(res)


@pytest.mark.parametrize("route,ok", [
    (["order_status", "refund", "FINISH"], True),
    (["order_status", "FINISH"], False),
])
def test_agents_requires_subset(route, ok):
    assert result(run({"agents": ["order_status", "refund"]}, route=route), "agents").passed is ok


def test_agents_exact_rejects_extra_agent():
    assert result(run({"agents_exact": ["refund"]}, route=["refund", "FINISH"]), "agents_exact").passed
    assert not result(run({"agents_exact": ["refund"]}, route=["refund", "order_status", "FINISH"]),
                      "agents_exact").passed


def test_tools_called_and_not_called():
    calls = [tool("refund", "check_return_eligibility")]
    assert result(run({"tools_called": ["check_return_eligibility"]}, tool_calls=calls), "tools_called").passed
    assert not result(run({"tools_called": ["initiate_refund"]}, tool_calls=calls), "tools_called").passed
    assert result(run({"tools_not_called": ["initiate_refund"]}, tool_calls=calls), "tools_not_called").passed
    assert not result(run({"tools_not_called": ["check_return_eligibility"]}, tool_calls=calls),
                      "tools_not_called").passed


def test_write_tool_only_refund():
    assert result(run({"write_tool_only_refund": True}, tool_calls=[tool("refund", "initiate_refund")]),
                  "write_tool_only_refund").passed
    assert not result(run({"write_tool_only_refund": True}, tool_calls=[tool("order_status", "initiate_refund")]),
                      "write_tool_only_refund").passed


def test_refund_state_status_and_absence():
    refunds = [{"order_id": 1008, "status": "completed"}]
    assert result(run({"refund_state": {1008: "completed", 1009: None}}, refunds=refunds), "refund_state").passed
    assert not result(run({"refund_state": {1008: "pending_approval"}}, refunds=refunds), "refund_state").passed
    assert not result(run({"refund_state": {1008: None}}, refunds=refunds), "refund_state").passed


def test_reviews_count():
    assert result(run({"reviews": 1}, reviews=[{"x": 1}]), "reviews").passed
    assert not result(run({"reviews": 0}, reviews=[{"x": 1}]), "reviews").passed


def test_flags():
    assert result(run({"escalated": True}, escalated=True), "escalated").passed
    assert not result(run({"escalated": False}, escalated=True), "escalated").passed
    assert result(run({"guard_blocked": True}, guard_blocked=True), "guard_blocked").passed
    assert not result(run({"guard_blocked": True}), "guard_blocked").passed


def test_reply_checks_are_warnings_not_failures():
    res = run({"reply_contains_any": ["shipped"], "reply_excludes": ["bob@example.com"]},
              replies=["Contact bob@example.com"])
    assert not result(res, "reply_contains_any").passed and not result(res, "reply_excludes").passed
    assert case_passed(res)  # warnings only
    ok = run({"reply_contains_any": ["SHIPPED"]}, replies=["Your order has shipped."])
    assert result(ok, "reply_contains_any").passed  # case-insensitive


def test_reply_must_contain_is_strict_and_checks_the_final_reply():
    res = run({"reply_must_contain": ["approved and processed"]},
              replies=["queued for review", "Your refund was Approved and Processed."])
    assert result(res, "reply_must_contain").passed  # last reply, case-insensitive
    bad = run({"reply_must_contain": ["approved and processed"]}, replies=["Your refund is queued for human review."])
    assert not result(bad, "reply_must_contain").passed and not case_passed(bad)  # strict: fails the case


def test_unknown_expect_key_is_rejected():
    with pytest.raises(ValueError, match="agent_exact"):
        run({"agent_exact": ["refund"]})
