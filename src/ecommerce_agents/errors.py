"""Classify model-call failures. Only Groq API errors are handled; bugs still raise."""
import groq

# Failures that may succeed later. The SDK already retried them before we see them.
TRANSIENT = (groq.RateLimitError, groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError)


def is_api_error(exc: BaseException) -> bool:
    return isinstance(exc, groq.APIError)


def is_transient(exc: BaseException) -> bool:
    """Rate limit, timeout, connection or server error (also 408/409/5xx status errors)."""
    if isinstance(exc, TRANSIENT):
        return True
    status = getattr(exc, "status_code", None) if isinstance(exc, groq.APIStatusError) else None
    return status in (408, 409) or (status is not None and status >= 500)


def is_tool_call_failure(exc: BaseException) -> bool:
    """Groq 400 `tool_use_failed`: the model produced a malformed tool call (or structured output).

    That is a sampling failure, not a bad request, so a fresh attempt usually succeeds. The SDK
    never retries 400s; the node wrapper retries this one code. Any other 400 stays permanent.
    """
    if not isinstance(exc, groq.BadRequestError):
        return False
    body = exc.body if isinstance(exc.body, dict) else {}
    code = getattr(exc, "code", None) or body.get("code") or (body.get("error") or {}).get("code")
    return code == "tool_use_failed" or "tool_use_failed" in str(exc)


def reason(exc: BaseException) -> str:
    """Short label for logs: 'transient:RateLimitError' or 'permanent:AuthenticationError'."""
    return f"{'transient' if is_transient(exc) else 'permanent'}:{type(exc).__name__}"
