"""Report/summary logic with a fake runner: no graph, no API calls."""
import json

import pytest

from ecommerce_agents.evals.case import EvalCase, Observation
from ecommerce_agents.evals.report import (evaluate, format_summary, run_cases, select_cases, summarize,
                                           write_report)

CASES = [
    EvalCase("a", "routing", 1, ["x"], expect={"agents_exact": ["order_status"]}),
    EvalCase("b", "routing", 1, ["x"], expect={"agents_exact": ["product_info"]}),
    EvalCase("c", "refund_policy", 1, ["x"], expect={"refund_state": {1008: "completed"},
                                                      "reply_contains_any": ["done"]}),
]


def fake_runner(case):
    """a passes, b routes wrongly, c has the right DB state but a reply missing the warned text."""
    if case.id == "a":
        return Observation("a", route=["order_status", "FINISH"])
    if case.id == "b":
        return Observation("b", route=["order_status", "FINISH"])
    return Observation("c", refunds=[{"order_id": 1008, "status": "completed"}], replies=["ok"])


def test_select_filters_and_rejects_typos():
    assert [c.id for c in select_cases(CASES, categories=["routing"])] == ["a", "b"]
    assert [c.id for c in select_cases(CASES, ids=["c"])] == ["c"]
    assert [c.id for c in select_cases(CASES, categories=["routing"], ids=["b"])] == ["b"]
    with pytest.raises(ValueError, match="category"):
        select_cases(CASES, categories=["routin"])
    with pytest.raises(ValueError, match="case id"):
        select_cases(CASES, ids=["zzz"])


def test_evaluate_record_shape():
    rec = evaluate(CASES[0], fake_runner(CASES[0]))
    assert rec["passed"] and rec["id"] == "a" and rec["judge"] is None
    assert {c["name"] for c in rec["checks"]} == {"no_error", "agents_exact"}
    json.dumps(rec)  # must be serialisable


def test_summary_rates_failures_and_warnings():
    records = run_cases(CASES, runner=fake_runner)
    s = summarize(records)
    assert s["cases"] == {"passed": 2, "total": 3, "rate": 0.667}
    assert s["by_category"]["routing"] == {"passed": 1, "total": 2, "rate": 0.5}
    assert s["by_category"]["refund_policy"]["rate"] == 1.0
    assert s["by_metric"]["routing"] == {"passed": 1, "total": 2, "rate": 0.5}
    assert s["by_metric"]["policy"]["rate"] == 1.0
    assert "reply_text" not in s["by_metric"]  # warnings are not strict metrics
    assert s["reply_warnings"] == 1  # case c's reply lacks "done" but still passes
    assert [f["id"] for f in s["failures"]] == ["b"]
    assert s["failures"][0]["failed_checks"][0]["name"] == "agents_exact"


def test_on_record_called_per_case():
    seen = []
    run_cases(CASES, runner=fake_runner, on_record=lambda r: seen.append(r["id"]))
    assert seen == ["a", "b", "c"]


def test_empty_run_has_no_division_error():
    s = summarize([])
    assert s["cases"]["rate"] is None and s["failures"] == []


def test_format_lists_failure_details():
    text = format_summary(summarize(run_cases(CASES, runner=fake_runner)))
    assert "cases passed" in text and "Failures:" in text and "agents_exact" in text and "66.7%" in text


def test_write_report_roundtrip(tmp_path):
    records = run_cases(CASES, runner=fake_runner)
    path = write_report(records, summarize(records), tmp_path / "reports", model="m")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.parent.name == "reports" and data["agent_model"] == "m"
    assert len(data["records"]) == 3 and data["summary"]["cases"]["total"] == 3
