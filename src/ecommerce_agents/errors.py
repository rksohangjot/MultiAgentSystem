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


def reason(exc: BaseException) -> str:
    """Short label for logs: 'transient:RateLimitError' or 'permanent:AuthenticationError'."""
    return f"{'transient' if is_transient(exc) else 'permanent'}:{type(exc).__name__}"
