"""Structured requests to an OpenAI-compatible model endpoint."""

import logging
import os
import time
from functools import cache

from openai import OpenAI
from pydantic import BaseModel

# OpenAI-compatible endpoint. Point OPENAI_BASE_URL at any local server.
MODEL = os.environ.get("LARS_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.environ.get("LARS_TEMPERATURE", 1.2))
# LARS_NO_THINK=1 disables reasoning on Qwen-style local models (~20x faster).
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}} if os.environ.get("LARS_NO_THINK") == "1" else None
log = logging.getLogger("lars.llm")


@cache
def client() -> OpenAI:
    return OpenAI(timeout=float(os.environ.get("LARS_LLM_TIMEOUT", 60)), max_retries=1)


def ask[T: BaseModel](prompt: str, schema: type[T], temperature: float = TEMPERATURE) -> T:
    """Send one prompt, get back a validated Pydantic object."""
    start = time.monotonic()
    res = client().chat.completions.parse(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        response_format=schema, temperature=temperature, extra_body=EXTRA,
    )
    message = res.choices[0].message
    parsed = message.parsed
    log.info("%s in %.2fs", schema.__name__, time.monotonic() - start)
    if parsed is None:
        raise ValueError(f"LLM returned no parseable {schema.__name__}: {message.content!r}")
    return parsed
