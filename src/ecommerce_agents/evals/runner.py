"""Run one eval case through the real graph, on a throwaway database, and record an Observation."""
import contextlib
import datetime as _dt
import tempfile
import uuid
from pathlib import Path

from langgraph.types import Command

from ecommerce_agents import db
from ecommerce_agents.chat_loop import new_message
from ecommerce_agents.evals.case import EvalCase, Observation
from ecommerce_agents.graph import build_graph
from ecommerce_agents.identity import get_customer
from ecommerce_agents.tools import refund_tools as rt


@contextlib.contextmanager
def sandbox(today: str):
    """Fresh seeded DB in a temp dir plus a pinned 'today', both restored on exit."""
    y, m, d = (int(p) for p in today.split("-"))

    class Frozen(_dt.date):
        @classmethod
        def today(cls):
            return cls(y, m, d)

    old_path, old_date = db.DB_PATH, rt.date
    with tempfile.TemporaryDirectory() as tmp:
        db.DB_PATH = Path(tmp) / "eval.db"
        rt.date = Frozen
        db.seed()
        try:
            yield
        finally:
            db.DB_PATH, rt.date = old_path, old_date


def _record_tools(update_messages, node, seen: set, calls: list, by_call_id: dict) -> None:
    """Collect tool calls and their results from a node's update, attributed to that node."""
    for m in update_messages:
        mid = getattr(m, "id", None)
        if mid in seen:  # agent nodes return their whole history; count each message once
            continue
        seen.add(mid)
        if m.type == "ai":
            for tc in getattr(m, "tool_calls", None) or []:
                entry = {"agent": node, "name": tc["name"], "args": tc["args"], "result": None}
                calls.append(entry)
                by_call_id[tc["id"]] = entry
        elif m.type == "tool" and m.tool_call_id in by_call_id:
            by_call_id[m.tool_call_id]["result"] = str(m.content)


def _run_turn(graph, graph_input, cfg, case: EvalCase, obs: Observation, seen, by_call_id, decisions) -> None:
    """Stream one customer turn; each approval pause is answered from the scripted decisions."""
    while True:
        paused = None
        for step in graph.stream(graph_input, cfg, stream_mode="updates"):
            for node, upd in step.items():
                if node == "__interrupt__":
                    paused = upd[0].value
                    continue
                upd = upd or {}
                if node == "supervisor" and upd.get("next_agent"):
                    obs.route.append(upd["next_agent"])
                _record_tools(upd.get("messages", []), node, seen, obs.tool_calls, by_call_id)
        if paused is None:
            return
        obs.reviews.append(paused)
        if not decisions:
            obs.error = f"case {case.id}: approval pause with no scripted reviewer decision"
            return
        graph_input = Command(resume=decisions.pop(0))


def run_case(case: EvalCase) -> Observation:
    """Execute every turn of `case` with a fresh graph and database. Never raises for agent
    behaviour; a crash inside the graph is recorded in `error` so one bad case can't stop a run."""
    obs = Observation(case_id=case.id)
    with sandbox(case.today):
        customer = get_customer(case.customer_id)
        cfg = {"configurable": {"thread_id": f"eval-{case.id}-{uuid.uuid4().hex[:8]}",
                                "customer_id": case.customer_id},
               "recursion_limit": 25}
        graph, seen, by_call_id, decisions = build_graph(), set(), {}, list(case.reviewer)
        try:
            for text in case.messages:
                _run_turn(graph, new_message(text, customer), cfg, case, obs, seen, by_call_id, decisions)
                if obs.error:
                    break
                values = graph.get_state(cfg).values
                ai = [m for m in values.get("messages", []) if m.type == "ai" and m.content]
                obs.replies.append(ai[-1].content if ai else "")
                obs.escalated = obs.escalated or bool(values.get("escalated"))
                obs.guard_blocked = obs.guard_blocked or bool(values.get("guard_blocked"))
                obs.guard_notice = values.get("guard_notice") or obs.guard_notice
        except Exception as exc:  # noqa: BLE001 - a crashing case is data, not a reason to abort the run
            obs.error = f"{type(exc).__name__}: {exc}"
        conn = db.get_conn()
        try:
            obs.refunds = [dict(r) for r in conn.execute("SELECT * FROM refunds ORDER BY id")]
        finally:
            conn.close()
    return obs
