from pydantic import BaseModel

from lars.llm import ask


class Strategies(BaseModel):
    strategies: list[str]


def propose_strategies(feed: str, n: int = 4) -> list[str]:
    """LLM proposes n distinct tweet-writing styles (the initial style levels)."""
    prompt = (
        f"Propose {n} distinct styles for writing tweets on {feed}. Each is one short "
        "sentence describing an angle or format (not a topic, mood, length or visual look)."
    )
    return ask(prompt, Strategies).strategies[:n]
