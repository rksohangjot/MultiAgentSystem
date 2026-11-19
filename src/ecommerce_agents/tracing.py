"""Optional Langfuse tracing.

Off unless both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, so tests and the
CLI work offline. The langfuse package is imported lazily and tracing must never break
a customer turn: setup failures are logged and tracing is skipped.
"""
import logging
import os

from dotenv import load_dotenv

from .guards import redact_pii

log = logging.getLogger(__name__)

_client = None


def tracing_enabled() -> bool:
    load_dotenv()
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def mask(data):
    """Second layer of PII protection: redact every string before it leaves the process.

    Langfuse calls this on trace inputs/outputs. The input guard already redacts customer
    messages, so this mostly catches tool output and model text. Recurses into dicts/lists.
    """
    if isinstance(data, str):
        return redact_pii(data).text
    if isinstance(data, dict):
        return {k: mask(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask(v) for v in data]
    return data


def _init_client():
    """Create the Langfuse client once. The callback handler picks it up as the global client."""
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(mask=mask)
    return _client


def callbacks() -> list:
    """LangChain callbacks for the run config: a Langfuse handler, or [] when tracing is off."""
    if not tracing_enabled():
        return []
    try:
        _init_client()
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        log.warning("Langfuse tracing could not be set up; continuing without it", exc_info=True)
        return []


def flush() -> None:
    """Send buffered traces. Call before the process exits or the last turn can be lost."""
    if _client is None:
        return
    try:
        _client.flush()
    except Exception:
        log.warning("Langfuse flush failed", exc_info=True)
