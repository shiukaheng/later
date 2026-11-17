import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(override=True)

# OpenAI-compatible endpoint. Point OPENAI_BASE_URL at any local server.
MODEL = os.environ.get("LARS_MODEL", "gpt-4o-mini")
client = OpenAI()
# LARS_NO_THINK=1 disables reasoning on Qwen-style local models (~20x faster).
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}} if os.environ.get("LARS_NO_THINK") == "1" else None


def ask[T: BaseModel](prompt: str, schema: type[T], temperature: float = 1.0) -> T:
    """Send one prompt, get back a validated Pydantic object."""
    res = client.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=schema,
        temperature=temperature,
        extra_body=EXTRA,
    )
    return res.choices[0].message.parsed
