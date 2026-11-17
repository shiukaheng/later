from pydantic import BaseModel

from lars.llm import ask


class Tweet(BaseModel):
    text: str


def generate_tweet(levels: dict[str, str], feed: str) -> str:
    """LLM writes one tweet following the chosen level of each dimension."""
    spec = "\n".join(f"- {dim}: {level}" for dim, level in levels.items())
    prompt = (
        f"Write one short, realistic tweet as it might appear on {feed}.\n"
        f"Follow every requirement strictly, but never mention or label them "
        f"(no hashtags or words naming the mood, style, etc.):\n{spec}"
    )
    return ask(prompt, Tweet).text
