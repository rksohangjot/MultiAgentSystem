"""Failure handling: SDK retry semantics we rely on, and error classification (no API calls)."""
import groq
import httpx
import pytest

from ecommerce_agents import errors, llm


def client_with(handler, max_retries=2):
    return groq.Groq(api_key="k", max_retries=max_retries, base_url="http://test",
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def ask(client):
    return client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])


OK = {"id": "1", "object": "chat.completion", "created": 0, "model": "m",
      "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]}
FAST = {"retry-after-ms": "1"}  # keep backoff tiny in tests


def counting(*statuses):
    calls = []

    def handler(request):
        status = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(status)
        return httpx.Response(status, json=OK if status == 200 else {"error": {"message": "x"}}, headers=FAST)
    return handler, calls


def test_429_then_success_is_retried():
    handler, calls = counting(429, 200)
    assert ask(client_with(handler)).choices[0].message.content == "ok"
    assert calls == [429, 200]


def test_always_429_gives_up_after_three_attempts():
    handler, calls = counting(429)
    with pytest.raises(groq.RateLimitError):
        ask(client_with(handler))
    assert len(calls) == 3


def test_server_error_is_retried():
    handler, calls = counting(503, 200)
    ask(client_with(handler))
    assert calls == [503, 200]


@pytest.mark.parametrize("status,exc", [(401, groq.AuthenticationError), (400, groq.BadRequestError),
                                        (403, groq.PermissionDeniedError)])
def test_auth_and_bad_request_are_never_retried(status, exc):
    handler, calls = counting(status)
    with pytest.raises(exc):
        ask(client_with(handler))
    assert calls == [status]


def test_get_llm_uses_explicit_retry_and_timeout(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    m = llm.get_llm()
    assert m.max_retries == 2 and m.request_timeout == 20.0
    monkeypatch.setenv("GROQ_MAX_RETRIES", "5")
    monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "7")
    m = llm.get_llm()
    assert m.max_retries == 5 and m.request_timeout == 7.0


# --- classification ------------------------------------------------------------------
def status_error(cls, status):
    req = httpx.Request("POST", "http://test")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


@pytest.mark.parametrize("exc,transient", [
    (status_error(groq.RateLimitError, 429), True),
    (status_error(groq.InternalServerError, 503), True),
    (groq.APITimeoutError(httpx.Request("POST", "http://test")), True),
    (groq.APIConnectionError(request=httpx.Request("POST", "http://test")), True),
    (status_error(groq.APIStatusError, 408), True),
    (status_error(groq.AuthenticationError, 401), False),
    (status_error(groq.BadRequestError, 400), False),
    (status_error(groq.PermissionDeniedError, 403), False),
])
def test_is_transient(exc, transient):
    assert errors.is_transient(exc) is transient and errors.is_api_error(exc)


def test_programming_errors_are_not_api_errors():
    assert not errors.is_api_error(KeyError("bug")) and not errors.is_transient(KeyError("bug"))


def test_reason_label():
    assert errors.reason(status_error(groq.RateLimitError, 429)) == "transient:RateLimitError"
    assert errors.reason(status_error(groq.AuthenticationError, 401)) == "permanent:AuthenticationError"


# --- graph: failures after retries become a human handoff ------------------------------------
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from ecommerce_agents import supervisor  # noqa: E402
from ecommerce_agents.graph import build_graph  # noqa: E402
from ecommerce_agents.resilience import HANDOFF  # noqa: E402
from tests.test_refund_agent import ScriptedModel, call  # noqa: E402
from tests.test_refund_graph import run, seeded, stub_models, stub_supervisor  # noqa: E402,F401


class RaisingModel(ScriptedModel):
    """Like ScriptedModel, but a script item that is an exception is raised instead of returned."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        item = self.script[min(self.i, len(self.script) - 1)]
        if isinstance(item, BaseException):
            self.i += 1
            raise item
        return super()._generate(messages, stop, run_manager, **kwargs)


RATE_LIMIT = status_error(groq.RateLimitError, 429)
AUTH = status_error(groq.AuthenticationError, 401)


def ai_texts(out):
    return [m.content for m in out["messages"] if m.type == "ai" and m.content]


def test_supervisor_failure_ends_turn_with_one_handoff(monkeypatch):
    def down(messages):
        raise RATE_LIMIT
    monkeypatch.setattr(supervisor, "decide", down)
    stub_models(monkeypatch)
    out = run("where is order 1008")
    assert ai_texts(out) == [HANDOFF] and out["escalated"] is True


def test_specialist_failure_ends_turn_without_asking_supervisor_again(monkeypatch):
    calls = []
    stub_supervisor(monkeypatch, ["order_status", "FINISH"])
    real = supervisor.decide
    monkeypatch.setattr(supervisor, "decide", lambda m: (calls.append(1), real(m))[1])
    stub_models(monkeypatch, order=RaisingModel(script=[RATE_LIMIT]))
    out = run("where is order 1008")
    assert ai_texts(out) == [HANDOFF] and len(calls) == 1  # supervisor ran once, not again


def test_refund_agent_failure_hands_off_and_writes_nothing(monkeypatch):
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    stub_models(monkeypatch, refund_model=RaisingModel(script=[status_error(groq.InternalServerError, 503)]))
    out = run("refund order 1008")
    assert ai_texts(out) == [HANDOFF]
    conn = __import__("ecommerce_agents.db", fromlist=["db"]).get_conn()
    try:
        assert conn.execute("SELECT COUNT(*) FROM refunds WHERE order_id = 1008").fetchone()[0] == 0
    finally:
        conn.close()


def test_permanent_error_also_hands_off_but_logs_error(monkeypatch, caplog):
    stub_supervisor(monkeypatch, ["order_status"])
    stub_models(monkeypatch, order=RaisingModel(script=[AUTH]))
    with caplog.at_level("WARNING"):
        out = run("where is order 1008")
    assert ai_texts(out) == [HANDOFF]
    assert any(r.levelname == "ERROR" and "permanent:AuthenticationError" in r.message for r in caplog.records)


def test_transient_error_logs_warning(monkeypatch, caplog):
    def down(messages):
        raise RATE_LIMIT
    monkeypatch.setattr(supervisor, "decide", down)
    with caplog.at_level("WARNING"):
        run("hi")
    assert any(r.levelname == "WARNING" and "transient:RateLimitError" in r.message for r in caplog.records)


def test_bugs_are_not_swallowed(monkeypatch):
    def bug(messages):
        raise KeyError("bug")
    monkeypatch.setattr(supervisor, "decide", bug)
    with pytest.raises(KeyError):
        run("hi")


def test_escalated_flag_resets_next_turn(monkeypatch):
    state = {"n": 0}

    def flaky(messages):
        state["n"] += 1
        if state["n"] == 1:
            raise RATE_LIMIT
        return supervisor.RouteDecision(next_agent="FINISH", reply="Hello!")
    monkeypatch.setattr(supervisor, "decide", flaky)
    g = build_graph()
    cfg = {"configurable": {"thread_id": "r", "customer_id": 4}, "recursion_limit": 25}
    from ecommerce_agents.chat_loop import new_message
    first = g.invoke(new_message("hi"), cfg)
    second = g.invoke(new_message("hi again"), cfg)
    assert first["escalated"] is True and second["escalated"] is False
    assert ai_texts(second)[-1] == "Hello!"


# --- gap fix: a refund queued before the run died still reaches human approval ---------------------
from langgraph.types import Command  # noqa: E402

from ecommerce_agents.tools import refund_tools as rt  # noqa: E402
from tests.test_refund_graph import CFG, frozen_today  # noqa: E402,F401


def start(monkeypatch, model):
    stub_supervisor(monkeypatch, ["refund", "FINISH"])
    stub_models(monkeypatch, refund_model=model)
    g = build_graph()
    return g, g.invoke({"messages": [HumanMessage("Refund order 1011")], "iterations": 0}, CFG)


def test_model_failure_after_queueing_refund_still_pauses_for_approval(monkeypatch, frozen_today):
    model = RaisingModel(script=[call("initiate_refund", {"order_id": 1011, "reason": "x"}, "b"), RATE_LIMIT])
    g, out = start(monkeypatch, model)
    payload = out["__interrupt__"][0].value  # paused at the approval node, not lost
    assert payload["order_id"] == 1011
    out = g.invoke(Command(resume={"approved": True, "note": "ok"}), CFG)
    assert rt.get_refund(payload["refund_id"])["status"] == "completed"
    texts = ai_texts(out)
    assert texts[-2] == HANDOFF and "approved" in texts[-1]
    assert texts.count(HANDOFF) == 1  # supervisor was not asked again after the outage
    assert g.get_state(CFG).next == ()


def test_loop_cap_after_queueing_refund_still_pauses_for_approval(monkeypatch, frozen_today):
    model = ScriptedModel(script=[call("initiate_refund", {"order_id": 1011, "reason": "x"}, "b"),
                                  call("check_return_eligibility", {"order_id": 1011}, "c")], loop=True)
    g, out = start(monkeypatch, model)
    assert out["__interrupt__"][0].value["order_id"] == 1011


def test_failure_with_nothing_queued_still_just_ends(monkeypatch, frozen_today):
    g, out = start(monkeypatch, RaisingModel(script=[RATE_LIMIT]))
    assert "__interrupt__" not in out and ai_texts(out) == [HANDOFF]


def test_pending_refund_for_customer_is_scoped_to_that_customer(frozen_today):
    rt.create_refund(1011, "x", rt.date(2026, 11, 13))  # customer 1's order, over $50 -> pending
    assert rt.pending_refund_for_customer(1) is not None
    assert rt.pending_refund_for_customer(4) is None and rt.pending_refund_for_customer(None) is None
