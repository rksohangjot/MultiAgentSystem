"""Groq 400 `tool_use_failed` (malformed tool call) is retried at node level; other errors are not.

Errors are produced by the real Groq SDK against a mock transport, so the classifier sees the same
exception shape (status, parsed body, code) as in production. No API calls.
"""
import groq
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from ecommerce_agents import errors, supervisor, tracing
from ecommerce_agents.graph import build_graph
from ecommerce_agents.resilience import HANDOFF, escalate_on_api_error
from ecommerce_agents.supervisor import RouteDecision
from ecommerce_agents.tools import refund_tools as rt
from tests.test_failures import RaisingModel, ai_texts, client_with
from tests.test_refund_agent import ScriptedModel, call
from tests.test_refund_graph import CFG, big_refund_model, frozen_today, seeded, stub_models, stub_supervisor  # noqa: F401

TOOL_USE_FAILED = {"error": {
    "message": "Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.",
    "type": "invalid_request_error", "code": "tool_use_failed",
    "failed_generation": "<function=get_order_status>{order_id: }</function>"}}
PLAIN_400 = {"error": {"message": "Invalid value for 'temperature'.", "type": "invalid_request_error"}}


def sdk_error(status, body):
    """The exception the real SDK raises for this response (never retried by the SDK itself)."""
    client = client_with(lambda request: httpx.Response(status, json=body), max_retries=0)
    with pytest.raises(groq.APIStatusError) as info:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
    return info.value


def tool_failure():
    return sdk_error(400, TOOL_USE_FAILED)


# --- classification ------------------------------------------------------------------------

def test_tool_use_failed_is_recognised_from_the_real_sdk_exception():
    exc = tool_failure()
    assert isinstance(exc, groq.BadRequestError) and errors.is_tool_call_failure(exc)
    assert not errors.is_transient(exc)  # still "permanent" for logging; only the node wrapper retries it


@pytest.mark.parametrize("status,body", [
    (400, PLAIN_400), (401, {"error": {"message": "bad key"}}), (429, {"error": {"message": "slow down"}}),
    (500, {"error": {"message": "oops"}}),
])
def test_other_api_errors_are_not_tool_call_failures(status, body):
    assert not errors.is_tool_call_failure(sdk_error(status, body))


def test_a_plain_bug_is_not_a_tool_call_failure():
    assert not errors.is_tool_call_failure(KeyError("tool_use_failed"))


# --- the wrapper -----------------------------------------------------------------------------

@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(tracing, "record_event", lambda name, **detail: seen.append((name, detail)))
    return seen


class Flaky:
    """A node that raises the given exceptions in order, then returns a normal update."""
    def __init__(self, *failures):
        self.failures, self.calls = list(failures), 0

    def __call__(self, state, config):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"messages": [AIMessage(content="fine", name="n")]}


def test_fail_once_then_succeed_gives_a_normal_reply(events):
    node = Flaky(tool_failure())
    out = escalate_on_api_error(node, "order_status")({"messages": []}, {})
    assert node.calls == 2 and out["messages"][0].content == "fine" and "escalated" not in out
    assert events == [("retry.tool_call_failed", {"node": "order_status", "attempt": 1})]


def test_always_failing_makes_three_attempts_then_hands_off(events):
    node = Flaky(*[tool_failure() for _ in range(10)])
    out = escalate_on_api_error(node, "supervisor")({"messages": []}, {})
    assert node.calls == 3  # 1 try + 2 retries
    assert out["messages"][0].content == HANDOFF and out["escalated"] is True and out["next_agent"] == "FINISH"
    assert [e[0] for e in events] == ["retry.tool_call_failed", "retry.tool_call_failed", "escalation.api_error"]
    assert [e[1].get("attempt") for e in events[:2]] == [1, 2]
    assert events[2][1]["reason"] == "permanent:BadRequestError"


@pytest.mark.parametrize("exc_factory", [
    lambda: sdk_error(400, PLAIN_400),
    lambda: sdk_error(401, {"error": {"message": "bad key"}}),
])
def test_other_permanent_errors_are_not_retried(exc_factory, events):
    node = Flaky(exc_factory())
    out = escalate_on_api_error(node, "refund")({"messages": []}, {})
    assert node.calls == 1 and out["escalated"] is True
    assert [e[0] for e in events] == ["escalation.api_error"]


def test_programming_errors_are_not_retried_or_swallowed(events):
    node = Flaky(KeyError("bug"))
    with pytest.raises(KeyError):
        escalate_on_api_error(node, "refund")({"messages": []}, {})
    assert node.calls == 1 and events == []


@pytest.mark.parametrize("setting,attempts", [("0", 1), ("1", 2), ("4", 5)])
def test_retry_count_is_configurable(monkeypatch, setting, attempts, events):
    monkeypatch.setenv("GROQ_TOOL_CALL_RETRIES", setting)
    node = Flaky(*[tool_failure() for _ in range(10)])
    escalate_on_api_error(node, "supervisor")({"messages": []}, {})
    assert node.calls == attempts


def test_retry_event_carries_categories_only(events):
    escalate_on_api_error(Flaky(tool_failure()), "product_info")({"messages": []}, {})
    (name, detail), = events
    assert set(detail) == {"node", "attempt"} and "failed_generation" not in str(detail)


# --- in the real graph -----------------------------------------------------------------------

def test_supervisor_retry_succeeds_and_does_not_burn_an_iteration(monkeypatch, events):
    calls = []

    def decide(messages):
        calls.append(1)
        if len(calls) == 1:
            raise tool_failure()
        return RouteDecision(next_agent="FINISH", reply="Hello!")
    monkeypatch.setattr(supervisor, "decide", decide)
    stub_models(monkeypatch)
    out = build_graph().invoke({"messages": [HumanMessage("hi")], "iterations": 0},
                               {"configurable": {"thread_id": "t", "customer_id": 4}, "recursion_limit": 25})
    assert ai_texts(out) == ["Hello!"] and not out.get("escalated")
    assert out["iterations"] == 1  # the failed attempt returned no update, so it was not counted


def test_specialist_retry_succeeds_with_a_normal_answer(monkeypatch):
    stub_supervisor(monkeypatch, ["order_status", "FINISH"])
    model = RaisingModel(script=[tool_failure(), AIMessage(content="Order 1008 was delivered.")])
    stub_models(monkeypatch, order=model)
    out = build_graph().invoke({"messages": [HumanMessage("where is order 1008")], "iterations": 0},
                               {"configurable": {"thread_id": "t", "customer_id": 4}, "recursion_limit": 25})
    assert ai_texts(out) == ["Order 1008 was delivered."] and not out.get("escalated")


def start(monkeypatch, model):
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    stub_models(monkeypatch, refund_model=model)
    g = build_graph()
    return g, g.invoke({"messages": [HumanMessage("Refund order 1011")], "iterations": 0}, CFG)


def test_refund_retried_after_queueing_still_reaches_approval(monkeypatch, frozen_today):
    # Attempt 1 queues the refund then the model's next call is malformed; the re-run finds the
    # pending row (initiate_refund is idempotent) and the approval node is still reached.
    model = RaisingModel(script=[
        call("initiate_refund", {"order_id": 1011, "reason": "x"}, "b"), tool_failure(),
        call("initiate_refund", {"order_id": 1011, "reason": "x"}, "c"),
        AIMessage(content="A human agent will review your refund."),
    ])
    g, out = start(monkeypatch, model)
    payload = out["__interrupt__"][0].value
    assert payload["order_id"] == 1011
    out = g.invoke(Command(resume={"approved": True, "note": "ok"}), CFG)
    assert rt.get_refund(payload["refund_id"])["status"] == "completed"
    assert HANDOFF not in ai_texts(out) and "approved" in ai_texts(out)[-1]


def test_refund_retries_exhausted_after_queueing_still_reaches_approval(monkeypatch, frozen_today):
    model = RaisingModel(script=[call("initiate_refund", {"order_id": 1011, "reason": "x"}, "b"),
                                 tool_failure(), tool_failure(), tool_failure()])
    g, out = start(monkeypatch, model)
    payload = out["__interrupt__"][0].value  # handoff happened, but the queued refund was not lost
    assert payload["order_id"] == 1011
    out = g.invoke(Command(resume={"approved": True, "note": "ok"}), CFG)
    assert ai_texts(out).count(HANDOFF) == 1
