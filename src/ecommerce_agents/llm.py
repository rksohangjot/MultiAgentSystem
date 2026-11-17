"""Groq model factory.

Retries live here, once, at the model level, so the supervisor and every agent get them.
The Groq SDK retries 408/409/429/5xx and connection errors with exponential backoff
(honouring Retry-After) and never retries 400/401/403/404. Tools are never retried.
"""
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MAX_RETRIES = 2        # retries after the first try => 3 attempts per model call
DEFAULT_TIMEOUT_SECONDS = 20.0  # per request


def get_llm(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=temperature,
        max_retries=int(os.getenv("GROQ_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
        request_timeout=float(os.getenv("GROQ_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
    )
