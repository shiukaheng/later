import logging
import os
import time

from openai import OpenAI
from pydantic import BaseModel

# OpenAI-compatible endpoint. Point OPENAI_BASE_URL at any local server.
MODEL = os.environ.get("LARS_MODEL", "gpt-4o-mini")
client = OpenAI(timeout=float(os.environ.get("LARS_LLM_TIMEOUT", 60)), max_retries=1)
# LARS_NO_THINK=1 disables reasoning on Qwen-style local models (~20x faster).
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}} if os.environ.get("LARS_NO_THINK") == "1" else None
log = logging.getLogger("lars.llm")


def ask[T: BaseModel](prompt: str, schema: type[T], temperature: float = 1.0) -> T:
    """Send one prompt, get back a validated Pydantic object."""
    start = time.monotonic()
    res = client.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=schema,
        temperature=temperature,
        extra_body=EXTRA,
    )
    parsed = res.choices[0].message.parsed
    log.info("%s in %.2fs (%d tokens)", schema.__name__, time.monotonic() - start, res.usage.completion_tokens)
    if parsed is None:
        raise ValueError(f"LLM returned no parseable {schema.__name__}: {res.choices[0].message.content!r}")
    return parsed
