"""Run the eval dataset against the live graph (calls the Groq API).

  PYTHONPATH=src uv run python scripts/run_evals.py                      # all cases
  PYTHONPATH=src uv run python scripts/run_evals.py --category routing   # one or more categories
  PYTHONPATH=src uv run python scripts/run_evals.py --case refund_small_auto --case route_greeting
  PYTHONPATH=src uv run python scripts/run_evals.py --list               # show the cases and exit

Writes data/eval_reports/eval-<timestamp>.json. Exit code 1 if any case fails.
This is not part of pytest: it costs API calls and the LLM is not deterministic.
"""
import argparse
import os
import sys
from pathlib import Path

from ecommerce_agents.evals.dataset import CASES
from ecommerce_agents.evals.report import format_summary, run_cases, select_cases, summarize, write_report

REPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_reports"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", action="append", help="only this category (repeatable)")
    ap.add_argument("--case", action="append", help="only this case id (repeatable)")
    ap.add_argument("--list", action="store_true", help="list the selected cases and exit")
    ap.add_argument("--out", type=Path, default=REPORT_DIR, help="report directory")
    args = ap.parse_args(argv)

    try:
        cases = select_cases(CASES, args.category, args.case)
    except ValueError as exc:
        ap.error(str(exc))
    if args.list:
        for c in cases:
            print(f"{c.category:<14} {c.id}")
        print(f"{len(cases)} case(s)")
        return 0
    if not os.getenv("GROQ_API_KEY"):
        from dotenv import load_dotenv  # llm.py loads it too; checked early for a clear message
        load_dotenv()
        if not os.getenv("GROQ_API_KEY"):
            print("GROQ_API_KEY is not set (copy .env.example to .env).", file=sys.stderr)
            return 2

    def show(rec):
        print(f"[{'PASS' if rec['passed'] else 'FAIL'}] {rec['category']:<14} {rec['id']}", flush=True)

    print(f"Running {len(cases)} case(s)...")
    records = run_cases(cases, on_record=show)
    summary = summarize(records)
    print(format_summary(summary))
    path = write_report(records, summary, args.out, os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    print(f"\nReport: {path}")
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
