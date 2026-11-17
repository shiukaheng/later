from pydantic import BaseModel

from lars.llm import ask


class Tweet(BaseModel):
    text: str


def generate_tweet(strategy: str, topic: str, audience: str) -> str:
    """LLM writes one tweet following the given strategy."""
    prompt = (
        f"Write one tweet (max 280 chars) about {topic} for {audience}.\n"
        f"Strategy: {strategy}"
    )
    return ask(prompt, Tweet).text
