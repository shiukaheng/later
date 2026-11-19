from pydantic import BaseModel

from lars.generation.llm import ask
from lars.generation.text import strip_markdown


# Style starts model-proposed; the other dimensions start with these options.
SEEDED = {
    "topic": [
        "tech and AI",
        "sports",
        "food and cooking",
        "pop culture, movies and music",
        "everyday life and relatable moments",
        "money, work and careers",
    ],
    "length": ["tiny: under 50 characters", "short: 50-100 characters", "medium: 100-160 characters"],
    "visual": [
        "plain text, no emojis or formatting",
        "one or two emojis",
        "emoji-heavy (5 or more emojis)",
        "line breaks or a short list",
        "ALL CAPS words for emphasis",
        "ends with hashtags",
    ],
    "mood": [
        "happy and upbeat",
        "ragebait: a provocative take designed to make people argue",
        "rant: frustrated and venting",
        "calm and matter-of-fact",
    ],
}


class Tweet(BaseModel):
    text: str


class Strategies(BaseModel):
    strategies: list[str]


def propose_strategies(feed: str, n: int = 4) -> list[str]:
    """Ask the model for the initial style options for a fresh profile."""
    prompt = (
        f"Propose {n} distinct styles for writing tweets on {feed}. Each is one short "
        "sentence describing an angle or format (not a topic, mood, length or visual look)."
    )
    return ask(prompt, Strategies).strategies[:n]


def generate_tweet(levels: dict[str, str], feed: str) -> str:
    """LLM writes one tweet following the chosen level of each dimension."""
    spec = "\n".join(f"- {dim}: {level}" for dim, level in levels.items())
    prompt = (
        f"Write one short, realistic tweet (at most 280 characters) as it might appear on {feed}.\n"
        f"Follow every requirement strictly, but never mention or label them "
        f"(do not name the mood, style, etc., in words or hashtags):\n{spec}\n"
        "Plain tweet text only. No markdown: no **bold**, *italics*, `code`, headings or "
        "[links](...). Twitter does not render markdown. Emojis and line breaks are fine. "
        "Keep the post evergreen; do not refer to current news events."
    )
    return strip_markdown(ask(prompt, Tweet).text)
