"""CLI loop tests: reviewer prompt, pause/resume driving, restart pickup (no API calls)."""
import pytest

from ecommerce_agents.chat_loop import cli_review, new_message, pending_review, resume_pending, stream_turn
from ecommerce_agents.graph import build_graph, sqlite_checkpointer
from ecommerce_agents.tools import refund_tools as rt
from tests.test_refund_graph import (CFG, big_refund_model, frozen_today, seeded,  # noqa: F401
                                     stub_models, stub_supervisor)


def answers(*items):
    it = iter(items)
    return lambda prompt: next(it)


def test_cli_review_approve_and_reject():
    payload = {"refund_id": 7, "order_id": 1011, "amount": 89.0, "reason": "changed mind"}
    shown = []
    assert cli_review(payload, answers("y", "fine"), shown.append) == {"approved": True, "note": "fine"}
    assert cli_review(payload, answers("n", "used"), shown.append) == {"approved": False, "note": "used"}
    assert "order 1011" in shown[0] and "$89.00" in shown[0]


def test_cli_review_reprompts_on_garbage():
    payload = {"refund_id": 1, "order_id": 1, "amount": 1.0, "reason": "x"}
    assert cli_review(payload, answers("maybe", "", "yes", ""), lambda s: None)["approved"] is True


def test_stream_turn_pauses_then_applies_reviewer_decision(monkeypatch, frozen_today):
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    stub_models(monkeypatch, refund_model=big_refund_model())
    g, lines = build_graph(), []
    seen = []
    stream_turn(g, new_message("Refund order 1011"), CFG,
                review=lambda p: seen.append(p) or {"approved": True, "note": ""}, emit=lines.append)
    assert seen[0]["order_id"] == 1011
    assert any("approved" in l for l in lines) and rt.get_refund(seen[0]["refund_id"])["status"] == "completed"


def test_pending_review_picked_up_after_restart(monkeypatch, frozen_today, tmp_path):
    f = tmp_path / "cp.db"
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    stub_models(monkeypatch, refund_model=big_refund_model())
    g1 = build_graph(sqlite_checkpointer(f))
    # Reviewer "walks away": the first process stops at the pause (review raises to end the loop).
    with pytest.raises(KeyboardInterrupt):
        stream_turn(g1, new_message("Refund order 1011"), CFG,
                    review=lambda p: (_ for _ in ()).throw(KeyboardInterrupt))
    stub_supervisor(monkeypatch, ["FINISH"])
    g2 = build_graph(sqlite_checkpointer(f))
    assert pending_review(g2, CFG)["order_id"] == 1011
    lines = []
    assert resume_pending(g2, CFG, lambda p: {"approved": False, "note": "no"}, lines.append) is True
    assert any("not approved" in l for l in lines)
    assert pending_review(g2, CFG) is None and resume_pending(g2, CFG, None) is False


@pytest.mark.llm
@pytest.mark.skipif(not __import__("os").getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
def test_live_over_50_refund_pauses_then_human_approves(frozen_today):
    g, seen = build_graph(), []
    stream_turn(g, new_message("Please refund order 1011, I changed my mind."), CFG,
                review=lambda p: seen.append(p) or {"approved": True, "note": "ok"}, emit=lambda s: None)
    assert len(seen) == 1 and seen[0]["order_id"] == 1011  # model reached the approval pause
    assert rt.get_refund(seen[0]["refund_id"])["status"] == "completed"
