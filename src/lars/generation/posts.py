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
        "instruction describing an angle or format (not a topic, mood, length or visual look). "
        "Aim for posts that feel like real Twitter/X: a recognizable situation, a concrete detail, "
        "or a specific opinion someone would actually share. For example, turn a small daily "
        "friction into a one-line observation that names the object and what went wrong. "
        "Avoid vague instructions like 'be relatable', generic engagement bait, and polished brand copy. "
        "Choose styles that would produce noticeably different posts even on the same topic."
    )
    return ask(prompt, Strategies).strategies[:n]


def generate_tweet(levels: dict[str, str], feed: str) -> str:
    """LLM writes one tweet following the chosen level of each dimension."""
    spec = "\n".join(f"- {dim}: {level}" for dim, level in levels.items())
    prompt = (
        f"Write one realistic tweet as it might appear on {feed}.\n"
        "Keep it concise: aim for one or two short sentences and stay under 280 characters.\n"
        f"Follow every requirement strictly, but never mention or label them "
        f"(do not name the mood, style, etc., in words or hashtags):\n{spec}\n"
        "Plain tweet text only. No markdown: no **bold**, *italics*, `code`, headings or "
        "[links](...). Twitter does not render markdown. Emojis and line breaks are fine. "
        "Keep the post evergreen; do not refer to current news events."
    )
    return strip_markdown(ask(prompt, Tweet).text)
