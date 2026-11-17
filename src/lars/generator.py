import os

from pydantic import BaseModel

from lars.llm import ask

# Higher = more varied tweets within each level combination.
TEMPERATURE = float(os.environ.get("LARS_TEMPERATURE", 1.2))


class Tweet(BaseModel):
    text: str


def generate_tweet(levels: dict[str, str], topic: str, audience: str) -> str:
    """LLM writes one tweet following the chosen level of each dimension."""
    spec = "\n".join(f"- {dim}: {level}" for dim, level in levels.items())
    prompt = (
        f"Write one short tweet about {topic} for {audience}.\n"
        f"Follow every requirement strictly:\n{spec}"
    )
    return ask(prompt, Tweet, temperature=TEMPERATURE).text
