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
