from pydantic import BaseModel

from lars.llm import ask


class Strategies(BaseModel):
    strategies: list[str]


def propose_strategies(topic: str, audience: str, n: int = 4) -> list[str]:
    """LLM proposes n distinct tweet-writing strategies (the bandit arms)."""
    prompt = (
        f"Propose {n} distinct strategies for writing tweets about {topic} "
        f"for {audience}. Each strategy is one short sentence describing style/angle."
    )
    return ask(prompt, Strategies).strategies[:n]
