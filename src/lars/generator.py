from pydantic import BaseModel

from lars.llm import ask


class Tweet(BaseModel):
    text: str


def generate_tweet(levels: dict[str, str], topic: str, audience: str) -> str:
    """LLM writes one tweet following the chosen level of each dimension."""
    spec = "\n".join(f"- {dim}: {level}" for dim, level in levels.items())
    prompt = (
        f"Write one tweet (max 280 chars) about {topic} for {audience}.\n"
        f"Follow every requirement strictly:\n{spec}"
    )
    return ask(prompt, Tweet).text
