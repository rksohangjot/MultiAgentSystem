"""Run cases, score them with the deterministic checks, summarise and write a JSON report.

Kept separate from scripts/run_evals.py so it can be unit tested with a fake runner (no LLM).
"""
import datetime as _dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from ecommerce_agents.evals.case import EvalCase, Observation
from ecommerce_agents.evals.checks import case_passed, check
from ecommerce_agents.evals.runner import run_case

# Check name -> the metric it feeds (the four DESIGN.md metrics, plus harness health).
METRIC_OF = {
    "agents": "routing", "agents_exact": "routing",
    "tools_called": "tool_correctness", "tools_not_called": "tool_correctness",
    "write_tool_only_refund": "tool_correctness",
    "refund_state": "policy", "reviews": "policy", "escalated": "policy", "guard_blocked": "policy",
    "reply_contains_any": "reply_text", "reply_excludes": "reply_text",
    "no_error": "harness",
}


def select_cases(cases: list[EvalCase], categories: list[str] | None = None,
                 ids: list[str] | None = None) -> list[EvalCase]:
    """Filter by category and/or id. An unknown category or id raises, so a typo is not an empty run."""
    known_cats, known_ids = {c.category for c in cases}, {c.id for c in cases}
    for cat in categories or []:
        if cat not in known_cats:
            raise ValueError(f"unknown category {cat!r}; choose from {sorted(known_cats)}")
    for cid in ids or []:
        if cid not in known_ids:
            raise ValueError(f"unknown case id {cid!r}")
    return [c for c in cases
            if (not categories or c.category in categories) and (not ids or c.id in ids)]


def evaluate(case: EvalCase, obs: Observation) -> dict:
    """One JSON-serialisable record: verdict, every check, and the full observation."""
    results = check(case, obs)
    return {
        "id": case.id, "category": case.category, "judge_requested": case.judge,
        "passed": case_passed(results),
        "checks": [r.to_dict() for r in results],
        "observation": obs.to_dict(),
        "judge": None,  # filled in by the LLM judge (a later step)
    }


def run_cases(cases: list[EvalCase], runner: Callable[[EvalCase], Observation] = run_case,
              on_record: Callable[[dict], None] | None = None) -> list[dict]:
    records = []
    for case in cases:
        record = evaluate(case, runner(case))
        records.append(record)
        if on_record:
            on_record(record)
    return records


def _rate(passed: int, total: int) -> dict:
    return {"passed": passed, "total": total, "rate": round(passed / total, 3) if total else None}


def summarize(records: list[dict]) -> dict:
    """Case pass rate overall and per category, strict-check pass rate per metric, and failures."""
    by_cat: dict[str, list[bool]] = defaultdict(list)
    by_metric: dict[str, list[bool]] = defaultdict(list)
    warnings = 0
    for rec in records:
        by_cat[rec["category"]].append(rec["passed"])
        for c in rec["checks"]:
            if c["strict"]:
                by_metric[METRIC_OF.get(c["name"], "other")].append(c["passed"])
            elif not c["passed"]:
                warnings += 1
    failures = [{"id": r["id"], "failed_checks": [{"name": c["name"], "detail": c["detail"]}
                                                    for c in r["checks"] if c["strict"] and not c["passed"]]}
                for r in records if not r["passed"]]
    return {
        "cases": _rate(sum(r["passed"] for r in records), len(records)),
        "by_category": {k: _rate(sum(v), len(v)) for k, v in sorted(by_cat.items())},
        "by_metric": {k: _rate(sum(v), len(v)) for k, v in sorted(by_metric.items())},
        "reply_warnings": warnings,
        "failures": failures,
    }


def _line(label: str, r: dict) -> str:
    pct = "n/a" if r["rate"] is None else f"{r['rate'] * 100:5.1f}%"
    return f"  {label:<18} {r['passed']:>3}/{r['total']:<3} {pct}"


def format_summary(summary: dict) -> str:
    out = ["", "== Eval summary ==", _line("cases passed", summary["cases"]), "By category:"]
    out += [_line(k, v) for k, v in summary["by_category"].items()]
    out.append("By metric (strict checks):")
    out += [_line(k, v) for k, v in summary["by_metric"].items()]
    out.append(f"Reply-text warnings (non-fatal): {summary['reply_warnings']}")
    if summary["failures"]:
        out.append("Failures:")
        for f in summary["failures"]:
            out.append(f"  {f['id']}")
            out += [f"    - {c['name']}: {c['detail']}" for c in f["failed_checks"]]
    return "\n".join(out)


def write_report(records: list[dict], summary: dict, out_dir: Path, model: str | None = None) -> Path:
    """Write <out_dir>/eval-<timestamp>.json with the summary and every per-case record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"eval-{stamp}.json"
    payload = {"created_at": _dt.datetime.now().isoformat(timespec="seconds"), "agent_model": model,
               "summary": summary, "records": records}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
