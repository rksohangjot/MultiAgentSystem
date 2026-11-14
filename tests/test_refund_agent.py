"""Refund agent tests with a scripted fake model (no API calls) + optional live tests."""
import os

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ecommerce_agents import db
from ecommerce_agents.agents import refund


class ScriptedModel(BaseChatModel):
    """Plays back AIMessages in order; if `loop`, repeats the last forever."""
    script: list
    loop: bool = False
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        idx = min(self.i, len(self.script) - 1) if self.loop else self.i
        msg = self.script[idx]
        if self.loop:  # unique tool-call ids so each loop turn is distinct
            msg = msg.model_copy(update={"id": f"m{self.i}"})
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


@pytest.fixture(autouse=True)
def seeded(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.seed(path)


def refunds():
    conn = db.get_conn()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM refunds ORDER BY id")]
    finally:
        conn.close()


def test_agent_tool_allowlist():
    names = [t.name for t in refund.REFUND_AGENT_TOOLS]
    assert names == ["get_order_status", "get_return_policy", "check_return_eligibility", "initiate_refund"]
    assert "search_products" not in names


def test_happy_path_check_then_refund(monkeypatch):
    model = ScriptedModel(script=[
        call("check_return_eligibility", {"order_id": 1008}, "a"),
        call("initiate_refund", {"order_id": 1008, "reason": "broken"}, "b"),
        AIMessage(content="Your refund for order 1008 has been issued."),
    ])
    monkeypatch.setattr(refund, "get_llm", lambda *a, **k: model)
    node = refund.build_refund_agent()
    out = node({"messages": [HumanMessage("Refund order 1008, it is broken")]}, {})
    assert out["messages"][-1].content.startswith("Your refund")
    assert [r["order_id"] for r in refunds()] == [1010, 1008]


def test_loop_cap_triggers_human_handoff(monkeypatch):
    model = ScriptedModel(script=[call("check_return_eligibility", {"order_id": 1008}, "x")], loop=True)
    monkeypatch.setattr(refund, "get_llm", lambda *a, **k: model)
    node = refund.build_refund_agent()
    out = node({"messages": [HumanMessage("refund 1008")]}, {"recursion_limit": 25})
    assert out["messages"][-1].content == refund.HANDOFF
    assert model.i <= refund.RECURSION_LIMIT  # capped by the agent's own limit, not the outer 25
    assert refunds() == [refunds()[0]] and len(refunds()) == 1  # nothing written


needs_key = pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")


@pytest.mark.llm
@needs_key
def test_live_explicit_refund_issues_refund():
    node = refund.build_refund_agent()
    node({"messages": [HumanMessage("Please refund order 1008, the case is cracked.")]}, {})
    assert 1008 in [r["order_id"] for r in refunds()]


@pytest.mark.llm
@needs_key
def test_live_question_does_not_issue_refund():
    node = refund.build_refund_agent()
    node({"messages": [HumanMessage("Am I able to return order 1008?")]}, {})
    assert 1008 not in [r["order_id"] for r in refunds()]


@pytest.mark.llm
@needs_key
def test_live_over_50_not_refunded():
    node = refund.build_refund_agent()
    node({"messages": [HumanMessage("Refund order 1011 please, I changed my mind.")]}, {})
    assert 1011 not in [r["order_id"] for r in refunds()]
